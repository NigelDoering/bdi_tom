"""
Trajectory Encoder - Transformer-based trajectory encoding with Node2Vec embeddings.

This module provides a transformer-based encoder for agent trajectories using
pre-computed Node2Vec embeddings instead of learned embeddings from scratch.
"""

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    """
    Encodes an agent's trajectory using a Transformer with pre-computed node embeddings.
    
    Instead of learning embeddings from scratch, this encoder uses pre-computed Node2Vec
    embeddings, ensuring a shared embedding space with the map encoder.
    
    Also incorporates the hour of day as trajectory-level context.
    """
    def __init__(self, node_emb_dim, hidden_dim, output_dim, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=50):
        """
        Args:
            node_emb_dim (int): Dimension of pre-computed node embeddings (from Node2Vec)
            hidden_dim (int): Hidden dimension for transformer feedforward network
            output_dim (int): Final output embedding dimension
            n_layers (int): Number of transformer encoder layers
            n_heads (int): Number of attention heads
            dropout (float): Dropout rate
            max_seq_len (int): Maximum sequence length for positional encoding
        """
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Positional encoding (learnable) - CRITICAL for sequence order!
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, node_emb_dim) * 0.02
        )
        
        # Hour embedding (24 hours in a day)
        self.hour_embedding = nn.Embedding(24, node_emb_dim)
        
        # Transformer encoder (processes node embedding sequences)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=node_emb_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection (combines trajectory embedding + hour embedding)
        self.output_proj = nn.Sequential(
            nn.Linear(node_emb_dim * 2, hidden_dim),  # *2 because we concatenate traj + hour
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, node_embeddings, hour, mask=None):
        """
        Forward pass through the trajectory encoder.
        
        Args:
            node_embeddings: (batch_size, seq_len, node_emb_dim) - Pre-computed node embeddings
            hour: (batch_size,) - Hour of day (0-23) when trajectory occurred
            mask: (batch_size, seq_len) - Mask for padding (1 for valid, 0 for padding)
            
        Returns:
            Encoded trajectory representation of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = node_embeddings.shape
        
        # Create attention mask for padding if not provided
        if mask is None:
            # Assume padding is all zeros
            mask = (node_embeddings.sum(dim=-1) != 0).float()
        
        # Add positional encoding to preserve sequence order
        # This is CRITICAL - without it, the model can't distinguish [A,B,C] from [C,B,A]!
        pos_enc = self.positional_encoding[:, :seq_len, :]  # (1, seq_len, node_emb_dim)
        node_embeddings_with_pos = node_embeddings + pos_enc  # (batch_size, seq_len, node_emb_dim)
        
        # Apply transformer
        transformer_out = self.transformer(
            node_embeddings_with_pos, 
            src_key_padding_mask=(mask == 0)
        )  # (batch_size, seq_len, node_emb_dim)
        
        # Global mean pooling (average over valid tokens only)
        mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        traj_emb = (transformer_out * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        # (batch_size, node_emb_dim)
        
        # Embed hour (trajectory-level temporal context)
        hour_emb = self.hour_embedding(hour)  # (batch_size, node_emb_dim)
        
        # Concatenate trajectory and hour embeddings
        combined = torch.cat([traj_emb, hour_emb], dim=-1)  # (batch_size, node_emb_dim * 2)
        
        # Project to final output dimension
        output = self.output_proj(combined)  # (batch_size, output_dim)
        
        return output
