"""
MINIMAL ENCODER EXTRACTION SCRIPT

Extract only the UnifiedEmbeddingPipeline from a PerNodeToMPredictor checkpoint
and freeze it for use as a frozen encoder in transfer learning.

Usage:
    python extract_encoder.py \
        --checkpoint /path/to/best_model.pt \
        --output /path/to/saved_encoder.pt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.training.train_per_node import PerNodeToMPredictor
from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline


def _infer_config_from_state_dict(encoder_state_dict: Dict, verbose: bool = True) -> Dict:
    """
    Infer encoder configuration from state dict keys.
    
    This is a fallback for old checkpoints that don't have config metadata.
    Extracts dimensions from embedding matrices.
    """
    config = {}
    
    # Extract num_nodes from node embedding matrix
    # node_embedding.weight has shape [num_nodes, node_embedding_dim]
    for key, tensor in encoder_state_dict.items():
        if key.endswith('node_embedding.weight') or key.endswith('node2vec_encoder.node_embedding.weight'):
            if tensor.dim() == 2:
                config['num_nodes'] = tensor.shape[0]
                config['node_embedding_dim'] = tensor.shape[1]
                break
    
    # Extract num_categories from category embedding
    # Look for the shape [num_categories, embedding_dim] pattern
    for key, tensor in encoder_state_dict.items():
        if 'category_embedding.weight' in key or 'category_preference.weight' in key:
            if tensor.dim() == 2:
                # For node2vec_encoder.category_embedding: shape is [num_categories, embedding_dim]
                if 'node2vec' in key and tensor.shape[0] < tensor.shape[1]:
                    config['num_categories'] = tensor.shape[0]
                    break
    
    # Set standard defaults
    config.setdefault('num_agents', 1)
    config.setdefault('node_embedding_dim', 64)
    config.setdefault('temporal_dim', 64)
    config.setdefault('agent_dim', 64)
    config.setdefault('fusion_dim', 128)
    config.setdefault('hidden_dim', 256)
    config.setdefault('num_categories', 7)
    config.setdefault('n_fusion_layers', 2)
    config.setdefault('n_heads', 4)
    config.setdefault('dropout', 0.1)
    config.setdefault('use_node2vec', True)
    config.setdefault('use_temporal', True)
    config.setdefault('use_agent', True)
    config.setdefault('use_modality_gating', True)
    config.setdefault('use_cross_attention', True)
    
    if verbose:
        print("⚠️  Using inferred config from state dict:")
        for key, value in sorted(config.items()):
            print(f"   {key}: {value}")
    
    return config


def extract_encoder_from_checkpoint(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> UnifiedEmbeddingPipeline:
    """
    Extract the UnifiedEmbeddingPipeline from a PerNodeToMPredictor checkpoint.
    
    Args:
        checkpoint_path: Path to the saved PerNodeToMPredictor checkpoint (.pt)
        output_path: Optional path to save the extracted encoder. If None, not saved.
        verbose: Print detailed information about extraction
    
    Returns:
        Frozen UnifiedEmbeddingPipeline ready for transfer learning
    
    Example:
        >>> encoder = extract_encoder_from_checkpoint("checkpoints/best_model.pt")
        >>> # encoder is now frozen and ready to use
    """
    
    # ========================================================================
    # STEP 1: LOAD CHECKPOINT
    # ========================================================================
    if verbose:
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # ========================================================================
    # STEP 2: EXTRACT ENCODER STATE DICT
    # ========================================================================
    if verbose:
        print("\n🔍 Inspecting checkpoint structure...")
    
    # The checkpoint typically contains 'model_state_dict' or direct state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        full_state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        full_state_dict = checkpoint['state_dict']
    else:
        full_state_dict = checkpoint
    
    # Extract only embedding_pipeline keys
    encoder_state_dict = {}
    embedding_keys = [k for k in full_state_dict.keys() if k.startswith('embedding_pipeline.')]
    
    if not embedding_keys:
        raise ValueError(
            f"❌ No 'embedding_pipeline' keys found in checkpoint!\n"
            f"Available keys: {list(full_state_dict.keys())[:10]}..."
        )
    
    # Remove the 'embedding_pipeline.' prefix from keys
    for key in embedding_keys:
        new_key = key.replace('embedding_pipeline.', '')
        encoder_state_dict[new_key] = full_state_dict[key]
    
    if verbose:
        print(f"✅ Extracted {len(encoder_state_dict)} encoder parameters")
    
    # ========================================================================
    # STEP 3: RECONSTRUCT ENCODER (requires config from checkpoint)
    # ========================================================================
    if verbose:
        print("\n🏗️  Reconstructing UnifiedEmbeddingPipeline...")
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    
    # Fallback: Try to infer config from encoder state dict if not present
    if not config:
        if verbose:
            print("⚠️  No config in checkpoint, attempting to infer from state dict...")
        config = _infer_config_from_state_dict(encoder_state_dict, verbose)
    
    if not config:
        raise ValueError(
            "❌ No config found in checkpoint and could not infer! "
            "Please re-train with the updated train_per_node.py that saves config."
        )
    
    # Build encoder with config
    encoder = UnifiedEmbeddingPipeline(
        num_nodes=config['num_nodes'],
        num_agents=config['num_agents'],
        num_categories=config.get('num_categories', 7),
        node_embedding_dim=config.get('node_embedding_dim', 64),
        temporal_dim=config.get('temporal_dim', 64),
        agent_dim=config.get('agent_dim', 64),
        fusion_dim=config.get('fusion_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        n_fusion_layers=config.get('n_fusion_layers', 2),
        n_heads=config.get('n_heads', 4),
        dropout=config.get('dropout', 0.1),
        use_node2vec=config.get('use_node2vec', True),
        use_temporal=config.get('use_temporal', True),
        use_agent=config.get('use_agent', True),
        use_modality_gating=config.get('use_modality_gating', True),
        use_cross_attention=config.get('use_cross_attention', True),
    )
    
    # Load state dict into encoder
    encoder.load_state_dict(encoder_state_dict)
    
    if verbose:
        print("✅ UnifiedEmbeddingPipeline reconstructed")
    
    # ========================================================================
    # STEP 4: FREEZE ENCODER
    # ========================================================================
    if verbose:
        print("\n❄️  Freezing encoder parameters...")
    
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Verify freezing
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in encoder.parameters())
    
    if verbose:
        print(f"✅ Encoder frozen!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,} (should be 0)")
    
    # ========================================================================
    # STEP 5: SAVE ENCODER IF REQUESTED
    # ========================================================================
    if output_path:
        if verbose:
            print(f"\n💾 Saving extracted encoder to: {output_path}")
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save encoder state dict + config
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'config': config,
            'frozen': True,
        }, output_path)
        
        if verbose:
            print(f"✅ Encoder saved!")
    
    return encoder


def load_frozen_encoder(checkpoint_path: str) -> UnifiedEmbeddingPipeline:
    """
    Quick load of a previously extracted and frozen encoder.
    
    Args:
        checkpoint_path: Path to the saved encoder checkpoint
    
    Returns:
        Frozen UnifiedEmbeddingPipeline
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    encoder = UnifiedEmbeddingPipeline(
        num_nodes=config['num_nodes'],
        num_agents=config['num_agents'],
        num_categories=config.get('num_categories', 7),
        node_embedding_dim=config.get('node_embedding_dim', 64),
        temporal_dim=config.get('temporal_dim', 64),
        agent_dim=config.get('agent_dim', 64),
        fusion_dim=config.get('fusion_dim', 128),
        hidden_dim=config.get('hidden_dim', 256),
        n_fusion_layers=config.get('n_fusion_layers', 2),
        n_heads=config.get('n_heads', 4),
        dropout=config.get('dropout', 0.1),
        use_node2vec=config.get('use_node2vec', True),
        use_temporal=config.get('use_temporal', True),
        use_agent=config.get('use_agent', True),
        use_modality_gating=config.get('use_modality_gating', True),
        use_cross_attention=config.get('use_cross_attention', True),
    )
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    # Ensure frozen
    for param in encoder.parameters():
        param.requires_grad = False
    
    return encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract UnifiedEmbeddingPipeline from PerNodeToMPredictor checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PerNodeToMPredictor checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save extracted encoder (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    try:
        encoder = extract_encoder_from_checkpoint(
            args.checkpoint,
            output_path=args.output,
            verbose=args.verbose,
        )
        print("\n" + "="*70)
        print("✨ Encoder extraction complete!")
        print("="*70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
