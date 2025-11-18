"""
Goal Prediction Model

This module implements a simple goal prediction model that takes the fused
trajectory + map embeddings and outputs a probability distribution over POI nodes.

Architecture:
    Fusion Embedding (from ToMGraphEncoder) 
    → Transformer Layer (optional refinement)
    → MLP Classifier
    → Softmax → Probability Distribution over POIs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GoalPredictionModel(nn.Module):
    """
    Simple goal prediction model that outputs probability distribution over POI nodes.
    
    Takes the fused embedding from ToMGraphEncoder and predicts which POI is the goal.
    """
    
    def __init__(
        self,
        fusion_encoder,
        num_poi_nodes,
        fusion_dim=64,
        hidden_dim=128,
        n_transformer_layers=1,
        n_heads=4,
        dropout=0.2
    ):
        """
        Initialize goal prediction model.
        
        Args:
            fusion_encoder: ToMGraphEncoder instance (trajectory + map fusion)
            num_poi_nodes (int): Number of POI nodes to predict over
            fusion_dim (int): Dimension of fusion embedding (default: 64)
            hidden_dim (int): Hidden dimension for transformer and MLP (default: 128)
            n_transformer_layers (int): Number of transformer layers (default: 1)
            n_heads (int): Number of attention heads (default: 4)
            dropout (float): Dropout rate (default: 0.2)
        """
        super().__init__()
        
        self.fusion_encoder = fusion_encoder
        self.num_poi_nodes = num_poi_nodes
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        
        # Project fusion embedding to hidden dimension
        self.input_proj = nn.Linear(fusion_dim, hidden_dim)
        
        # Transformer layer for further refinement (optional)
        if n_transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        else:
            self.transformer = None
        
        # Classification head: MLP that outputs logits for each POI
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_poi_nodes)
        )
    
    def forward(self, trajectory_data, graph_data, return_logits=False):
        """
        Forward pass through the complete model.
        
        Args:
            trajectory_data (dict): Dictionary containing trajectory information
                - 'node_embeddings': (batch_size, seq_len, node_emb_dim) - Node2Vec embeddings
                - 'hour': (batch_size,) - Hour of day
                - 'mask': (batch_size, seq_len) - Padding mask (optional)
            graph_data (dict): Dictionary containing world graph information
                - 'node_embeddings': (num_nodes, node_emb_dim) - Node2Vec embeddings for all nodes
                - 'edge_index': (2, num_edges) - Edge connectivity
            return_logits (bool): If True, return logits instead of probabilities
        
        Returns:
            torch.Tensor: 
                - If return_logits=False: (batch_size, num_poi_nodes) - Probability distribution
                - If return_logits=True: (batch_size, num_poi_nodes) - Logits
        """
        # Get fused embedding from trajectory + map
        fused_emb = self.fusion_encoder(trajectory_data, graph_data)  # (batch_size, fusion_dim)
        
        # Project to hidden dimension
        x = self.input_proj(fused_emb)  # (batch_size, hidden_dim)
        x = F.relu(x)
        
        # Apply transformer layer (optional refinement)
        if self.transformer is not None:
            # Transformer expects (batch, seq_len, hidden_dim), so unsqueeze
            x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            x = self.transformer(x)  # (batch_size, 1, hidden_dim)
            x = x.squeeze(1)  # (batch_size, hidden_dim)
        
        # Classify: output logits for each POI
        logits = self.classifier(x)  # (batch_size, num_poi_nodes)
        
        if return_logits:
            return logits
        else:
            # Return probability distribution
            return F.softmax(logits, dim=-1)
    
    def predict_top_k(self, trajectory_data, graph_data, k=5):
        """
        Predict top-k most likely goal POIs.
        
        Args:
            trajectory_data (dict): Trajectory data
            graph_data (dict): Graph data
            k (int): Number of top predictions to return (default: 5)
        
        Returns:
            tuple: (top_k_probs, top_k_indices)
                - top_k_probs: (batch_size, k) - Probabilities of top-k POIs
                - top_k_indices: (batch_size, k) - Indices of top-k POIs
        """
        probs = self.forward(trajectory_data, graph_data, return_logits=False)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        return top_k_probs, top_k_indices
    
