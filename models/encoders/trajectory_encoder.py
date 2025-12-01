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
    
    Also incorporates:
    - Hour of day as trajectory-level temporal context
    - Agent ID to capture agent-specific preferences and behavioral patterns
    """
    def __init__(self, node_emb_dim, hidden_dim, output_dim, num_agents, n_layers=2, n_heads=4, dropout=0.1, max_seq_len=50):
        """
        Args:
            node_emb_dim (int): Dimension of pre-computed node embeddings (from Node2Vec)
            hidden_dim (int): Hidden dimension for transformer feedforward network
            output_dim (int): Final output embedding dimension
            num_agents (int): Total number of unique agents in the dataset
            n_layers (int): Number of transformer encoder layers
            n_heads (int): Number of attention heads
            dropout (float): Dropout rate
            max_seq_len (int): Maximum sequence length for positional encoding
        """
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_agents = num_agents
        
        # Positional encoding (learnable) - CRITICAL for sequence order!
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, node_emb_dim) * 0.02
        )
        
        # Hour embedding (24 hours in a day)
        self.hour_embedding = nn.Embedding(24, node_emb_dim)
        
        # Agent embedding - learns agent-specific behavioral patterns
        # Each agent gets a unique learned vector capturing their preferences
        self.agent_embedding = nn.Embedding(num_agents, node_emb_dim)
        
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
        
        # Output projection (combines trajectory embedding + hour embedding + agent embedding)
        self.output_proj = nn.Sequential(
            nn.Linear(node_emb_dim * 3, hidden_dim),  # *3 because we concatenate traj + hour + agent
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, node_embeddings, hour, agent_id, mask=None):
        """
        Forward pass through the trajectory encoder.
        
        Args:
            node_embeddings: (batch_size, seq_len, node_emb_dim) - Pre-computed node embeddings
            hour: (batch_size,) - Hour of day (0-23) when trajectory occurred
            agent_id: (batch_size,) - Agent ID (integer indices) for each trajectory
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
        
        # Embed agent ID (agent-specific behavioral patterns)
        agent_emb = self.agent_embedding(agent_id)  # (batch_size, node_emb_dim)
        
        # Concatenate trajectory, hour, and agent embeddings
        combined = torch.cat([traj_emb, hour_emb, agent_emb], dim=-1)  # (batch_size, node_emb_dim * 3)
        
        # Project to final output dimension
        output = self.output_proj(combined)  # (batch_size, output_dim)
        
        return output
