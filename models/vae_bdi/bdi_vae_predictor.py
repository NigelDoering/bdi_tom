"""
BDI-VAE Goal Prediction Model

This module implements the BDI (Belief-Desire-Intention) hierarchical VAE architecture
for goal prediction. The fusion embedding from ToMGraphEncoder flows through three
sequential Variational Autoencoders representing the BDI hierarchy:

Architecture:
    Fusion Embedding (from ToMGraphEncoder)
    → Belief VAE (encodes agent's understanding of environment)
    → Desire VAE (encodes agent's goals/preferences)
    → Intention VAE (encodes agent's action plans)
    → MLP Classifier
    → Softmax → Probability Distribution over POIs

Each VAE learns a latent representation at its level of the BDI hierarchy,
with gradual refinement from high-level beliefs to concrete intentions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VariationalAutoencoder(nn.Module):
    """
    Single Variational Autoencoder (VAE) module.
    
    Encodes input to latent distribution (mean, logvar) and samples from it.
    The reparameterization trick allows backpropagation through the stochastic sampling.
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dim=None, dropout=0.2):
        """
        Initialize VAE.
        
        Args:
            input_dim (int): Input dimension
            latent_dim (int): Latent space dimension
            hidden_dim (int): Hidden layer dimension (default: same as input_dim)
            dropout (float): Dropout rate (default: 0.2)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: input → hidden → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: latent → hidden → output (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, input_dim)
        
        Returns:
            Tuple of (mu, logvar) where:
                - mu: Mean of latent distribution (batch_size, latent_dim)
                - logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu (torch.Tensor): Mean (batch_size, latent_dim)
            logvar (torch.Tensor): Log variance (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Sampled latent vector (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent vector to reconstruction.
        
        Args:
            z (torch.Tensor): Latent vector (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Reconstructed input (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x, return_params=False):
        """
        Forward pass through VAE.
        
        Args:
            x (torch.Tensor): Input (batch_size, input_dim)
            return_params (bool): If True, return (z, mu, logvar, recon)
                                 If False, return only z (latent encoding)
        
        Returns:
            If return_params=True: Tuple of (z, mu, logvar, recon)
            If return_params=False: z (latent encoding)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        if return_params:
            recon = self.decode(z)
            return z, mu, logvar, recon
        else:
            return z
    
    def compute_loss(self, x, recon, mu, logvar, kl_weight=1.0):
        """
        Compute VAE loss = Reconstruction Loss + KL Divergence.
        
        Args:
            x (torch.Tensor): Original input (batch_size, input_dim)
            recon (torch.Tensor): Reconstructed input (batch_size, input_dim)
            mu (torch.Tensor): Latent mean (batch_size, latent_dim)
            logvar (torch.Tensor): Latent log variance (batch_size, latent_dim)
            kl_weight (float): Weight for KL divergence term (default: 1.0)
        
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence: KL(N(mu, sigma) || N(0, 1))
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


class BDIVAEPredictor(nn.Module):
    """
    BDI (Belief-Desire-Intention) Hierarchical VAE for Goal Prediction.
    
    This model implements a hierarchical BDI architecture:
    
    Architecture:
        Fusion Embedding (64-dim)
            ↓ ↓ (parallel)
            ├─→ Belief VAE (64 → 48 latent)
            └─→ Desire VAE (64 → 32 latent)
                    ↓
            Concat [fusion + belief + desire] = 144-dim
                    ↓
            Intention VAE (144 → 24 latent)
                    ↓
            MLP Classifier (24 → num_POIs)
    
    Flow:
    1. Belief VAE: Encodes fusion embedding → belief latent (48-dim)
    2. Desire VAE: Encodes fusion embedding → desire latent (32-dim) [parallel with Belief]
    3. Intention VAE: Encodes [fusion + belief + desire] → intention latent (24-dim)
    4. Classifier: Maps intention latent → goal predictions
    
    Reconstruction targets:
    - Belief VAE reconstructs fusion embedding
    - Desire VAE reconstructs fusion embedding
    - Intention VAE reconstructs concatenated [fusion + belief + desire]
    """
    
    def __init__(
        self,
        fusion_encoder,
        num_poi_nodes,
        fusion_dim=64,
        belief_latent_dim=48,
        desire_latent_dim=32,
        intention_latent_dim=24,
        hidden_dim=128,
        dropout=0.2,
        kl_weight_belief=1.0,
        kl_weight_desire=1.0,
        kl_weight_intention=1.0
    ):
        """
        Initialize BDI-VAE Predictor.
        
        Args:
            fusion_encoder: ToMGraphEncoder instance (trajectory + map fusion)
            num_poi_nodes (int): Number of POI nodes to predict over
            fusion_dim (int): Dimension of fusion embedding (default: 64)
            belief_latent_dim (int): Latent dimension for Belief VAE (default: 48)
            desire_latent_dim (int): Latent dimension for Desire VAE (default: 32)
            intention_latent_dim (int): Latent dimension for Intention VAE (default: 24)
            hidden_dim (int): Hidden dimension for VAEs and MLP (default: 128)
            dropout (float): Dropout rate (default: 0.2)
            kl_weight_belief (float): KL weight for Belief VAE (default: 1.0)
            kl_weight_desire (float): KL weight for Desire VAE (default: 1.0)
            kl_weight_intention (float): KL weight for Intention VAE (default: 1.0)
        """
        super().__init__()
        
        self.fusion_encoder = fusion_encoder
        self.num_poi_nodes = num_poi_nodes
        self.fusion_dim = fusion_dim
        
        # KL divergence weights for each VAE level
        self.kl_weight_belief = kl_weight_belief
        self.kl_weight_desire = kl_weight_desire
        self.kl_weight_intention = kl_weight_intention
        
        # Belief VAE: Processes fusion embedding
        # Input: fusion_dim → Latent: belief_latent_dim
        self.belief_vae = VariationalAutoencoder(
            input_dim=fusion_dim,
            latent_dim=belief_latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Desire VAE: ALSO processes fusion embedding (parallel with Belief)
        # Input: fusion_dim → Latent: desire_latent_dim
        self.desire_vae = VariationalAutoencoder(
            input_dim=fusion_dim,
            latent_dim=desire_latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Intention VAE: Processes fusion + belief + desire concatenated
        # Input: (fusion_dim + belief_latent_dim + desire_latent_dim) → Latent: intention_latent_dim
        intention_input_dim = fusion_dim + belief_latent_dim + desire_latent_dim
        self.intention_vae = VariationalAutoencoder(
            input_dim=intention_input_dim,
            latent_dim=intention_latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # MLP Classifier: Maps intention latent to POI predictions
        # Input: intention_latent_dim → Output: num_poi_nodes
        self.classifier = nn.Sequential(
            nn.Linear(intention_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_poi_nodes)
        )
    
    def forward(self, trajectory_data, graph_data, return_logits=False, return_vae_params=False):
        """
        Forward pass through the complete BDI-VAE model.
        
        Args:
            trajectory_data (dict): Dictionary containing trajectory information
                - 'node_ids': (batch_size, seq_len) - Node IDs in trajectory
                - 'hour': (batch_size,) - Hour of day
                - 'mask': (batch_size, seq_len) - Padding mask
            graph_data (dict): Dictionary containing world graph information
                - 'x': (num_nodes, node_feat_dim) - Node features
                - 'edge_index': (2, num_edges) - Edge connectivity
            return_logits (bool): If True, return logits instead of probabilities
            return_vae_params (bool): If True, return VAE parameters for loss computation
        
        Returns:
            If return_vae_params=False:
                torch.Tensor: Predictions (batch_size, num_poi_nodes)
                    - Logits if return_logits=True
                    - Probabilities if return_logits=False
            
            If return_vae_params=True:
                Tuple of (predictions, vae_params) where vae_params is a dict containing:
                    - 'belief_z', 'belief_mu', 'belief_logvar', 'belief_recon'
                    - 'desire_z', 'desire_mu', 'desire_logvar', 'desire_recon'
                    - 'intention_z', 'intention_mu', 'intention_logvar', 'intention_recon'
        """
        # Get fused embedding from trajectory + map
        fusion_emb = self.fusion_encoder(trajectory_data, graph_data)  # (batch_size, fusion_dim)
        
        # === BDI Hierarchy ===
        
        # 1. Belief VAE: Encode fusion embedding → belief latent
        if return_vae_params:
            belief_z, belief_mu, belief_logvar, belief_recon = self.belief_vae(
                fusion_emb, return_params=True
            )
        else:
            belief_z = self.belief_vae(fusion_emb, return_params=False)
        
        # 2. Desire VAE: ALSO encode fusion embedding → desire latent (parallel with Belief)
        if return_vae_params:
            desire_z, desire_mu, desire_logvar, desire_recon = self.desire_vae(
                fusion_emb, return_params=True
            )
        else:
            desire_z = self.desire_vae(fusion_emb, return_params=False)
        
        # 3. Intention VAE: Encode fusion + belief + desire → intention latent
        intention_input = torch.cat([fusion_emb, belief_z, desire_z], dim=-1)  # Concatenate all three
        if return_vae_params:
            intention_z, intention_mu, intention_logvar, intention_recon = self.intention_vae(
                intention_input, return_params=True
            )
        else:
            intention_z = self.intention_vae(intention_input, return_params=False)
        
        # 4. Classifier: Map intention latent → POI predictions
        logits = self.classifier(intention_z)  # (batch_size, num_poi_nodes)
        
        # Prepare return values
        if return_logits:
            predictions = logits
        else:
            predictions = F.softmax(logits, dim=-1)
        
        if return_vae_params:
            vae_params = {
                'fusion_emb': fusion_emb,
                'belief_z': belief_z,
                'belief_mu': belief_mu,
                'belief_logvar': belief_logvar,
                'belief_recon': belief_recon,
                'desire_z': desire_z,
                'desire_mu': desire_mu,
                'desire_logvar': desire_logvar,
                'desire_recon': desire_recon,
                'intention_input': intention_input,  # Save the concatenated input
                'intention_z': intention_z,
                'intention_mu': intention_mu,
                'intention_logvar': intention_logvar,
                'intention_recon': intention_recon
            }
            return predictions, vae_params
        else:
            return predictions
    
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
    
    def compute_loss(self, trajectory_data, graph_data, true_goal_indices):
        """
        Compute total loss = Classification Loss + VAE Reconstruction Losses.
        
        The total loss includes:
        1. Cross-entropy loss for goal classification
        2. Reconstruction + KL losses for each VAE (Belief, Desire, Intention)
        
        Args:
            trajectory_data (dict): Trajectory data
            graph_data (dict): Graph data
            true_goal_indices (torch.Tensor): (batch_size,) - True goal node indices
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains:
                - 'classification_loss': Cross-entropy for goal prediction
                - 'belief_loss', 'belief_recon', 'belief_kl': Belief VAE losses
                - 'desire_loss', 'desire_recon', 'desire_kl': Desire VAE losses
                - 'intention_loss', 'intention_recon', 'intention_kl': Intention VAE losses
                - 'total_vae_loss': Sum of all VAE losses
        """
        # Forward pass with VAE parameters
        logits, vae_params = self.forward(
            trajectory_data, graph_data, 
            return_logits=True, 
            return_vae_params=True
        )
        
        # 1. Classification loss
        classification_loss = F.cross_entropy(logits, true_goal_indices)
        
        # 2. Belief VAE loss - reconstructs fusion embedding
        belief_loss, belief_recon, belief_kl = self.belief_vae.compute_loss(
            vae_params['fusion_emb'],
            vae_params['belief_recon'],
            vae_params['belief_mu'],
            vae_params['belief_logvar'],
            kl_weight=self.kl_weight_belief
        )
        
        # 3. Desire VAE loss - ALSO reconstructs fusion embedding
        desire_loss, desire_recon, desire_kl = self.desire_vae.compute_loss(
            vae_params['fusion_emb'],
            vae_params['desire_recon'],
            vae_params['desire_mu'],
            vae_params['desire_logvar'],
            kl_weight=self.kl_weight_desire
        )
        
        # 4. Intention VAE loss - reconstructs concatenated (fusion + belief + desire)
        intention_loss, intention_recon, intention_kl = self.intention_vae.compute_loss(
            vae_params['intention_input'],  # The concatenated input
            vae_params['intention_recon'],
            vae_params['intention_mu'],
            vae_params['intention_logvar'],
            kl_weight=self.kl_weight_intention
        )
        
        # Total VAE loss
        total_vae_loss = belief_loss + desire_loss + intention_loss
        
        # Total loss
        total_loss = classification_loss + total_vae_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'classification_loss': classification_loss.item(),
            'belief_loss': belief_loss.item(),
            'belief_recon': belief_recon.item(),
            'belief_kl': belief_kl.item(),
            'desire_loss': desire_loss.item(),
            'desire_recon': desire_recon.item(),
            'desire_kl': desire_kl.item(),
            'intention_loss': intention_loss.item(),
            'intention_recon': intention_recon.item(),
            'intention_kl': intention_kl.item(),
            'total_vae_loss': total_vae_loss.item()
        }
        
        return total_loss, loss_dict


# -----------------------------
# Testing
# -----------------------------
def main():
    """Test the BDI-VAE Model with sample data."""
    print("=" * 80)
    print("Testing BDI-VAE Goal Prediction Model")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load world graph
    print("\n1. Loading world graph...")
    from graph_controller.world_graph import WorldGraph
    from models.encoders.map_encoder import GraphDataPreparator
    import networkx as nx
    
    graph_path = "data/processed/ucsd_walk_full.graphml"
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        return
    
    G = nx.read_graphml(graph_path)
    world_graph = WorldGraph(G)
    num_nodes = len(world_graph.G.nodes())
    num_poi_nodes = len(world_graph.poi_nodes)
    print(f"   ✓ Loaded graph: {num_nodes} total nodes, {num_poi_nodes} POI nodes")
    
    # Prepare graph data
    graph_prep = GraphDataPreparator(world_graph)
    graph_data = graph_prep.prepare_graph_data()
    print(f"   ✓ Graph features: {graph_data['x'].shape}")
    
    # Load trajectory data
    print("\n2. Loading trajectory data...")
    from models.encoders.trajectory_encoder import TrajectoryDataPreparator
    
    traj_path = "data/simulation_data/run_8/trajectories/all_trajectories.json"
    if not os.path.exists(traj_path):
        print(f"Error: Trajectory file not found at {traj_path}")
        return
    
    with open(traj_path, 'r') as f:
        trajectories = json.load(f)
    
    # Create node mapping
    node_list = list(world_graph.G.nodes())
    node_to_idx = {node: idx + 1 for idx, node in enumerate(node_list)}  # 0 reserved for padding
    node_to_idx['<PAD>'] = 0
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Create POI mapping (only POI nodes)
    poi_node_to_idx = {node: idx for idx, node in enumerate(world_graph.poi_nodes)}
    idx_to_poi = {idx: node for node, idx in poi_node_to_idx.items()}
    
    # Prepare sample trajectories
    traj_prep = TrajectoryDataPreparator(node_to_idx)
    agent_ids = list(trajectories.keys())[:3]
    sample_trajs = [trajectories[agent_id][0] for agent_id in agent_ids]
    trajectory_data = traj_prep.prepare_batch(sample_trajs, max_seq_len=50)
    print(f"   ✓ Loaded {len(sample_trajs)} trajectories")
    
    # Get true goal indices in POI space
    true_goals = []
    for traj in sample_trajs:
        goal_node = traj['goal_node']
        if goal_node in poi_node_to_idx:
            true_goals.append(poi_node_to_idx[goal_node])
        else:
            print(f"   Warning: Goal node {goal_node} not in POI list, using 0")
            true_goals.append(0)
    true_goal_indices = torch.tensor(true_goals, dtype=torch.long)
    
    print(f"   ✓ True goals: {[idx_to_poi[g] for g in true_goals]}")
    
    # Initialize fusion encoder
    print("\n3. Initializing ToM Fusion Encoder...")
    from models.encoders.fusion_encoder import ToMGraphEncoder
    
    fusion_encoder = ToMGraphEncoder(
        num_nodes=num_nodes + 1,  # +1 for padding
        graph_node_feat_dim=12,
        traj_node_emb_dim=32,
        hidden_dim=128,
        output_dim=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1
    )
    print(f"   ✓ Fusion encoder initialized (output_dim=64)")
    
    # Initialize BDI-VAE predictor
    print("\n4. Initializing BDI-VAE Predictor...")
    bdi_vae_predictor = BDIVAEPredictor(
        fusion_encoder=fusion_encoder,
        num_poi_nodes=num_poi_nodes,
        fusion_dim=64,
        belief_latent_dim=48,
        desire_latent_dim=32,
        intention_latent_dim=24,
        hidden_dim=128,
        dropout=0.2,
        kl_weight_belief=1.0,
        kl_weight_desire=1.0,
        kl_weight_intention=1.0
    )
    
    total_params = sum(p.numel() for p in bdi_vae_predictor.parameters())
    print(f"   ✓ BDI-VAE predictor initialized")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Predicting over {num_poi_nodes} POI nodes")
    print(f"   - Architecture: Fusion(64) → Belief(48) → Desire(32) → Intention(24) → Classifier")
    
    # Move to device
    bdi_vae_predictor = bdi_vae_predictor.to(device)
    for key in trajectory_data:
        trajectory_data[key] = trajectory_data[key].to(device)
    for key in graph_data:
        graph_data[key] = graph_data[key].to(device)
    true_goal_indices = true_goal_indices.to(device)
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    bdi_vae_predictor.eval()
    with torch.no_grad():
        # Get probability distribution
        probs = bdi_vae_predictor(trajectory_data, graph_data)
        print(f"   ✓ Output shape: {probs.shape} (batch_size={probs.shape[0]}, num_poi_nodes={probs.shape[1]})")
        
        # Verify it's a valid probability distribution
        prob_sums = probs.sum(dim=-1)
        print(f"   ✓ Probability sums: {prob_sums.tolist()} (should be ~1.0)")
        
        # Check all probabilities are in [0, 1]
        assert torch.all((probs >= 0) & (probs <= 1)), "Probabilities not in [0, 1]"
        print(f"   ✓ All probabilities in [0, 1]")
        
        # Get top-5 predictions for each trajectory
        print(f"\n6. Top-5 Goal Predictions:")
        top_k_probs, top_k_indices = bdi_vae_predictor.predict_top_k(trajectory_data, graph_data, k=5)
        
        for i, agent_id in enumerate(agent_ids):
            print(f"\n   Agent {agent_id}:")
            print(f"   True goal: {idx_to_poi[true_goals[i]]}")
            print(f"   Predictions:")
            for j in range(5):
                pred_idx = int(top_k_indices[i, j].item())
                pred_prob = top_k_probs[i, j].item()
                pred_node = idx_to_poi[pred_idx]
                marker = " ✓" if pred_idx == true_goals[i] else ""
                print(f"     {j+1}. {pred_node[:30]:30s} - {pred_prob:.4f}{marker}")
    
    # Test loss computation
    print(f"\n7. Testing loss computation...")
    bdi_vae_predictor.train()  # Need training mode for VAE
    total_loss, loss_dict = bdi_vae_predictor.compute_loss(trajectory_data, graph_data, true_goal_indices)
    print(f"   ✓ Total loss: {total_loss.item():.4f}")
    print(f"   ✓ Classification loss: {loss_dict['classification_loss']:.4f}")
    print(f"   ✓ Total VAE loss: {loss_dict['total_vae_loss']:.4f}")
    print(f"   - Belief VAE: loss={loss_dict['belief_loss']:.4f}, recon={loss_dict['belief_recon']:.4f}, kl={loss_dict['belief_kl']:.4f}")
    print(f"   - Desire VAE: loss={loss_dict['desire_loss']:.4f}, recon={loss_dict['desire_recon']:.4f}, kl={loss_dict['desire_kl']:.4f}")
    print(f"   - Intention VAE: loss={loss_dict['intention_loss']:.4f}, recon={loss_dict['intention_recon']:.4f}, kl={loss_dict['intention_kl']:.4f}")
    
    # Test with logits
    print(f"\n8. Testing logit output...")
    with torch.no_grad():
        logits = bdi_vae_predictor(trajectory_data, graph_data, return_logits=True)
        print(f"   ✓ Logits shape: {logits.shape}")
        print(f"   ✓ Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    # Test VAE parameters return
    print(f"\n9. Testing VAE parameters return...")
    with torch.no_grad():
        preds, vae_params = bdi_vae_predictor(
            trajectory_data, graph_data, 
            return_logits=False, 
            return_vae_params=True
        )
        print(f"   ✓ Returned {len(vae_params)} parameter tensors")
        print(f"   - Belief latent: {vae_params['belief_z'].shape}")
        print(f"   - Desire latent: {vae_params['desire_z'].shape}")
        print(f"   - Intention latent: {vae_params['intention_z'].shape}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
    print("\nModel Summary:")
    print(f"  - Input: Trajectory (variable length) + World Graph")
    print(f"  - Fusion Encoder: Combines trajectory + map → 64-dim embedding")
    print(f"  - Belief VAE: 64 → 48 (processes fusion embedding)")
    print(f"  - Desire VAE: 64 → 32 (processes fusion embedding, parallel with Belief)")
    print(f"  - Intention VAE: 144 → 24 (processes fusion + belief + desire)")
    print(f"  - Classifier: 3-layer MLP (24 → {num_poi_nodes} POIs)")
    print(f"  - Total parameters: {total_params:,}")
    print(f"\nArchitecture Flow:")
    print(f"  Fusion (64) → Belief VAE → belief (48)")
    print(f"               ↓")
    print(f"  Fusion (64) → Desire VAE → desire (32)")
    print(f"               ↓")
    print(f"  [Fusion + Belief + Desire] (144) → Intention VAE → intention (24)")
    print(f"               ↓")
    print(f"  Intention (24) → MLP → Predictions")
    print(f"\nLoss Components:")
    print(f"  - Classification: Cross-entropy on POI predictions")
    print(f"  - Belief VAE: Reconstructs fusion embedding")
    print(f"  - Desire VAE: Reconstructs fusion embedding")
    print(f"  - Intention VAE: Reconstructs [fusion + belief + desire]")


if __name__ == "__main__":
    main()
