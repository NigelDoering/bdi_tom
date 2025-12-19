"""
LSTM Goal Prediction Model

A baseline version of the transformer goal predictor:
Fusion Encoder → LSTM → Classifier → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LSTMGoalPredictionModel(nn.Module):
    """
    Baseline LSTM model for predicting goal POIs.
    
    Architecture:
        Fusion Encoder (trajectory + map data)
        → Linear projection to hidden dim
        → LSTM (1 layer)
        → MLP Classifier
        → Softmax probabilities
    """

    # ALL OF THIS IS JUST STO SET IT UP, THESE SELF ATTRIBUTES/MODEL WILL BE USED IN THE FORWARD FUNCTION
    def __init__(
        self,
        fusion_encoder,
        num_poi_nodes,
        fusion_dim=64,
        hidden_dim=128,
        lstm_layers=1,
        dropout=0.2
    ):
        """
        Initialize goal prediction model.
        
        Args:
            fusion_encoder: ToMGraphEncoder instance (trajectory + map fusion)
            num_poi_nodes (int): Number of POI nodes to predict over
            fusion_dim (int): Dimension of fusion embedding (default: 64)
            hidden_dim (int): Hidden dimension for transformer and MLP (default: 128)
            lstm_layers (int): Number of transformer layers (default: 1)
            dropout (float): Dropout rate (default: 0.2)
        """
        super().__init__()

        # TAKES TRAJ, NODE EMBEDDINGS, TIME OF DAY, GRAPH STRUCTURE, AND FUSES THEM INTO 1 SUMMMARY VECTOR
        # BUT THERE ARE WAY TOO MANY AGENTS, SO WE PROCESS THEM IN BATCHES OF (32, 64)
        self.fusion_encoder = fusion_encoder
        self.num_poi_nodes = num_poi_nodes
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim

        # TLDR BATCH SIZE (32, 64) -> (32, 128) ADD HIDDEN DIMENSIONS BECAUSE TRANSFORMER LIKES WORKING WITH MORE DIMENSIONS
        # Project fusion emb to LSTM hidden dimension
        self.input_proj = nn.Linear(fusion_dim, hidden_dim)

        # THIS IS THE CHANGED PART. LSTM INSTEAD OF TRANSFORMER
        # LSTM that processes the fused embedding
        # NOTE: we treat fused_emb as a sequence of length 1 (like transformer)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=(dropout if lstm_layers > 1 else 0.0)
        )

        # After the transformer/LSTM finishes processing the fused embedding, otput is a final vector like this: (32, 128)
        # It represents “Everything the model knows about what the agent is doing and where they might be going.”
        # Given this vector, which POI (goal node) is the agent heading toward? WE NEED TO REDUCE THE SHAPE A BIT SO SOFTMAX CAN RETURN IT
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), # reduce layers to compact info
            nn.ReLU(), # introduces non-linearity, makes sophisticated decisons
            nn.Dropout(dropout), # randomly makes some neurons 0 to prevent overfitting when training so it doesnt rely too much on a single traj
            nn.Linear(hidden_dim // 2, num_poi_nodes) # 
        )


    # THE POINT OF THIS FUNCTION IS TO NOT PROCESS MUCH, BUT TO CALL THE ATTRIBUTES THAT ALREADY HAD THE WORK PUT INTO IT, AND THEN USE IT
    # TO RETURN A SOFTMAX
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
        # STEP 1: Get fused embedding (batch, fusion_dim)
        # The fusion encoder combines the features into one summary vector of size (fusion_dim = 64)
        # "Here is my understanding of what the agent is doing + the environment they are in."
        fused_emb = self.fusion_encoder(trajectory_data, graph_data)

        # STEP 2: Project up to hidden dim
        # LSTM expects input size = hidden_dim (128)
        # More dimensions = more expressive power
        # ReLU adds non-linear transformation (lets model learn more complex patterns)
        # "Let me rewrite this 64-dimensional summary into 128-dimensional space so the LSTM has more room to think."
        x = self.input_proj(fused_emb)
        x = F.relu(x)

        # STEP 3: LSTM expects (batch, seq_len, hidden_dim)
        # PREPARE SO LSTM CAN TAKE IT. RESHAPE FROM (batch, 128) -> (batch, 1, 128)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)

        # STEP 4: Get output of LSTM on each timestep (we only care about the hidden state h_n, and specifically the LAST one)
        lstm_output, (h_n, c_n) = self.lstm(x)

        # h_n: SHAPE = (num_layers, batch, hidden_dim) -> (1, batch, 128)
        # → take last layer because that is the MOST UPDATED MEMORY/ONE AFTER THE LSTM PROCESS (batch, 128)
        h_last = h_n[-1]  # (batch, hidden_dim)

        # STEP3: classifier -> return softmax
        # computes scores for each POI in the map
        logits = self.classifier(h_last)

        if return_logits:
            return logits
        else:
            return F.softmax(logits, dim=-1) # turns something like [2.2, 1.4, -3.1, 0.7] into [0.55, 0.22, 0.01, 0.22]

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
