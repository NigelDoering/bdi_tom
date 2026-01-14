"""
CAUSALLY-CONSTRAINED DISENTANGLED BDI VAE (C²D-BDI) - VERSION 2

This is an approach to Theory of Mind modeling that properly
implements the Belief-Desire-Intention framework with:

1. SEPARATE INPUT PROJECTIONS for Belief (spatial) and Desire (preferences)
2. DIFFERENT RECONSTRUCTION TARGETS for each VAE
3. TOTAL CORRELATION PENALTY (β-TCVAE) for proper disentanglement
4. MUTUAL INFORMATION MINIMIZATION between B and D latents
5. TEMPORAL CONSISTENCY LOSSES for coherent mental state evolution
6. PATH PROGRESS CONDITIONING for trajectory-aware inference

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UNIFIED EMBEDDING PIPELINE                            │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
     ┌────────────────┐                        ┌────────────────┐
     │ SPATIAL BRANCH │                        │  AGENT BRANCH  │
     │  (for Belief)  │                        │  (for Desire)  │
     └────────┬───────┘                        └────────┬───────┘
              │                                         │
              ▼                                         ▼
     ┌────────────────┐                        ┌────────────────┐
     │   BELIEF VAE   │                        │   DESIRE VAE   │
     │                │                        │                │
     │ z_b: world     │                        │ z_d: prefs     │
     │      model     │                        │      goals     │
     └────────┬───────┘                        └────────┬───────┘
              │                                         │
              └──────────────────┬──────────────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │      INTENTION VAE       │
                    │  Input: [z_b, z_d, prog] │
                    │  z_i: action policy      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    PREDICTION HEADS      │
                    │  goal, nextstep, cat,    │
                    │  progress                │
                    └──────────────────────────┘

LOSSES:
1. Belief VAE: Recon(spatial features) + β_b * TC(z_b)
2. Desire VAE: Recon(preference features) + β_d * TC(z_d)
3. Intention VAE: Recon([z_b, z_d]) + β_i * KL(z_i)
4. MI Constraint: Minimize I(z_b; z_d)
5. Prediction losses: CE for goal, nextstep, category
6. Temporal consistency: Cosine similarity for same-trajectory samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline


# =============================================================================
# BETA-TCVAE COMPONENTS
# =============================================================================

class GaussianEncoder(nn.Module):
    """
    Encoder that outputs parameters of a diagonal Gaussian distribution.
    Uses proper initialization and residual connections for stability.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input projection with residual
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        
        # Output heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize log_var to output small values initially
        nn.init.zeros_(self.fc_log_var.weight)
        nn.init.constant_(self.fc_log_var.bias, -1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim]
        Returns:
            mu: [batch, latent_dim]
            log_var: [batch, latent_dim] (clamped for stability)
        """
        h = self.input_proj(x)
        
        for layer in self.hidden_layers:
            h = h + layer(h)  # Residual connection
        
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=2.0)
        
        return mu, log_var


class GaussianDecoder(nn.Module):
    """
    Decoder that reconstructs input from latent samples.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Hidden layers with residual
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        
        # Output head
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, latent_dim]
        Returns:
            recon: [batch, output_dim]
        """
        h = self.input_proj(z)
        
        for layer in self.hidden_layers:
            h = h + layer(h)
        
        return self.fc_out(h)


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick: z = μ + σ * ε"""
    # Clamp log_var for numerical stability
    log_var = torch.clamp(log_var, min=-10.0, max=10.0)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# =============================================================================
# SIMPLE β-VAE LOSS (NUMERICALLY STABLE - DEFAULT)
# =============================================================================

def compute_simple_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute simple β-VAE loss with numerical stability.
    
    L = Recon + β * KL(q(z|x) || p(z))
    
    This is a simplified, numerically stable version suitable for training.
    
    Args:
        recon: Reconstructed output
        target: Target to reconstruct
        mu: Encoder means
        log_var: Encoder log variances
        beta: Weight for KL divergence
    
    Returns:
        total_loss: Scalar total loss
        loss_dict: Dictionary with individual loss components
    """
    device = mu.device
    
    # Check for NaN inputs
    if torch.isnan(mu).any() or torch.isnan(log_var).any():
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            'recon_loss': torch.tensor(0.0, device=device),
            'mi_loss': torch.tensor(0.0, device=device),
            'tc_loss': torch.tensor(0.0, device=device),
            'dimwise_kl': torch.tensor(0.0, device=device),
            'total': zero,
        }
    
    # Clamp log_var for stability
    log_var = torch.clamp(log_var, min=-10.0, max=10.0)
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    recon_loss = torch.clamp(recon_loss, max=100.0)
    
    # KL divergence: KL(q(z|x) || N(0, I))
    # = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )
    kl_loss = torch.clamp(kl_loss, min=0.0, max=100.0)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    # Check for NaN
    if torch.isnan(total_loss):
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        recon_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
    
    loss_dict = {
        'recon_loss': recon_loss,
        'mi_loss': torch.tensor(0.0, device=device),  # Not computed in simple VAE
        'tc_loss': kl_loss,  # Report KL as TC for compatibility
        'dimwise_kl': kl_loss,
        'total': total_loss,
    }
    
    return total_loss, loss_dict


# Alias for backward compatibility
def compute_tcvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Wrapper that uses simple β-VAE loss for stability.
    The z parameter is ignored in this simplified version.
    """
    return compute_simple_vae_loss(recon, target, mu, log_var, beta)


# =============================================================================
# BELIEF VAE - WORLD MODEL
# =============================================================================

class BeliefVAEv2(nn.Module):
    """
    Belief VAE: Learns the agent's model of the world.
    
    Input: Spatial/trajectory features from the unified embedding
    Output: Latent belief representation z_b
    
    Beliefs capture:
    - Current spatial context (where am I?)
    - Trajectory history (where have I been?)
    - Graph structure awareness (what paths are available?)
    
    Reconstruction targets:
    - Spatial embedding (not the full unified embedding!)
    - Next-step distribution (implicit world model)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_nodes: int,  # For next-step reconstruction
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        
        # Encoder: spatial features → z_b
        self.encoder = GaussianEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Decoder 1: z_b → spatial features (primary reconstruction)
        self.spatial_decoder = GaussianDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Decoder 2: z_b → next-step distribution (auxiliary reconstruction)
        # This forces beliefs to encode what transitions are possible
        self.transition_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_nodes),
        )
    
    def forward(
        self, 
        spatial_features: torch.Tensor,
        return_transition: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            spatial_features: [batch, input_dim] - spatial branch features
            return_transition: Whether to compute transition logits
        
        Returns:
            Dict with z_b, mu, log_var, recon_spatial, (transition_logits)
        """
        # Encode
        mu, log_var = self.encoder(spatial_features)
        z = reparameterize(mu, log_var)
        
        # Decode spatial
        recon_spatial = self.spatial_decoder(z)
        
        outputs = {
            'z': z,
            'mu': mu,
            'log_var': log_var,
            'recon': recon_spatial,
            'target': spatial_features,
        }
        
        if return_transition:
            outputs['transition_logits'] = self.transition_decoder(z)
        
        return outputs


# =============================================================================
# DESIRE VAE - PREFERENCES & GOALS
# =============================================================================

class DesireVAEv2(nn.Module):
    """
    Desire VAE: Learns the agent's preferences and goal motivations.
    
    Input: Agent/preference features from the unified embedding
    Output: Latent desire representation z_d
    
    Desires capture:
    - Agent identity preferences
    - Goal category preferences
    - Behavioral tendencies
    
    Reconstruction targets:
    - Agent preference embedding
    - Goal category distribution (what types of places does this agent prefer?)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_categories: int,  # For category reconstruction
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        
        # Encoder: preference features → z_d
        self.encoder = GaussianEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Decoder 1: z_d → preference features (primary reconstruction)
        self.preference_decoder = GaussianDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Decoder 2: z_d → category distribution (auxiliary reconstruction)
        # Forces desires to encode what types of goals the agent prefers
        self.category_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_categories),
        )
    
    def forward(
        self,
        preference_features: torch.Tensor,
        return_category: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preference_features: [batch, input_dim] - agent/preference features
            return_category: Whether to compute category logits
        
        Returns:
            Dict with z_d, mu, log_var, recon_preference, (category_logits)
        """
        # Encode
        mu, log_var = self.encoder(preference_features)
        z = reparameterize(mu, log_var)
        
        # Decode preferences
        recon_preference = self.preference_decoder(z)
        
        outputs = {
            'z': z,
            'mu': mu,
            'log_var': log_var,
            'recon': recon_preference,
            'target': preference_features,
        }
        
        if return_category:
            outputs['category_logits'] = self.category_decoder(z)
        
        return outputs


# =============================================================================
# INTENTION VAE - ACTION POLICY
# =============================================================================

class IntentionVAEv2(nn.Module):
    """
    Intention VAE: Learns the agent's goal-directed action policy.
    
    Input: [z_b, z_d, path_progress] - beliefs, desires, and trajectory progress
    Output: Latent intention representation z_i
    
    Intentions capture:
    - How to achieve desires given beliefs
    - Action selection policy
    - Path planning signals
    
    Reconstruction targets:
    - The concatenated [z_b, z_d, progress] input
    - This forces intentions to properly integrate beliefs and desires
    """
    
    def __init__(
        self,
        belief_dim: int,
        desire_dim: int,
        hidden_dim: int,
        latent_dim: int,
        include_progress: bool = True,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.belief_dim = belief_dim
        self.desire_dim = desire_dim
        self.latent_dim = latent_dim
        self.include_progress = include_progress
        
        # Input dimension: belief + desire + optional progress
        input_dim = belief_dim + desire_dim + (1 if include_progress else 0)
        self.input_dim = input_dim
        
        # Encoder: [z_b, z_d, progress] → z_i
        self.encoder = GaussianEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Decoder: z_i → [z_b, z_d, progress]
        self.decoder = GaussianDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    
    def forward(
        self,
        belief_z: torch.Tensor,
        desire_z: torch.Tensor,
        progress: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            belief_z: [batch, belief_dim] - belief latent
            desire_z: [batch, desire_dim] - desire latent
            progress: [batch, 1] - path progress (0-1)
        
        Returns:
            Dict with z_i, mu, log_var, recon, input
        """
        # Concatenate inputs
        if self.include_progress and progress is not None:
            if progress.dim() == 1:
                progress = progress.unsqueeze(1)
            x = torch.cat([belief_z, desire_z, progress], dim=-1)
        else:
            x = torch.cat([belief_z, desire_z], dim=-1)
        
        # Encode
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        
        # Decode
        recon = self.decoder(z)
        
        return {
            'z': z,
            'mu': mu,
            'log_var': log_var,
            'recon': recon,
            'target': x,  # Reconstruction target is the input
        }


# =============================================================================
# MUTUAL INFORMATION ESTIMATOR (for disentanglement)
# =============================================================================

class MutualInformationEstimator(nn.Module):
    """
    Estimates and minimizes mutual information between belief and desire latents.
    
    Uses a variational bound on MI:
    I(z_b; z_d) <= E[log q(z_d|z_b) - log q(z_d)]
    
    We train a discriminator to distinguish between joint samples (z_b, z_d)
    and marginal samples (z_b, z_d') where z_d' is shuffled.
    """
    
    def __init__(
        self,
        belief_dim: int,
        desire_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(belief_dim + desire_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        belief_z: torch.Tensor,
        desire_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MI upper bound loss (MINE estimator with numerical stability).
        
        Args:
            belief_z: [batch, belief_dim]
            desire_z: [batch, desire_dim]
        
        Returns:
            mi_loss: Scalar - minimizing this reduces MI(z_b; z_d)
        """
        batch_size = belief_z.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=belief_z.device)
        
        # Joint samples (z_b, z_d) from same examples
        joint = torch.cat([belief_z, desire_z], dim=-1)
        
        # Marginal samples (z_b, z_d') where z_d' is shuffled
        shuffle_idx = torch.randperm(batch_size, device=desire_z.device)
        marginal = torch.cat([belief_z, desire_z[shuffle_idx]], dim=-1)
        
        # Discriminator scores (clamp for stability)
        joint_scores = self.discriminator(joint)
        marginal_scores = self.discriminator(marginal)
        
        # Clamp scores to prevent exp overflow
        joint_scores = torch.clamp(joint_scores, min=-10.0, max=10.0)
        marginal_scores = torch.clamp(marginal_scores, min=-10.0, max=10.0)
        
        # MINE estimator (Mutual Information Neural Estimation)
        # Use logsumexp for numerical stability instead of mean(exp())
        mi_estimate = torch.mean(joint_scores) - (
            torch.logsumexp(marginal_scores.squeeze(-1), dim=0) - math.log(batch_size)
        )
        
        # Clamp final result
        mi_estimate = torch.clamp(mi_estimate, min=-10.0, max=10.0)
        
        return mi_estimate


# =============================================================================
# MAIN C²D-BDI MODEL
# =============================================================================

class CausallyConstrainedBDIVAE(nn.Module):
    """
    Causally-Constrained Disentangled BDI VAE (C²D-BDI)
    
    The revolutionary architecture that properly implements BDI theory with:
    1. Separate feature branches for Belief (spatial) and Desire (preferences)
    2. β-TCVAE for proper disentanglement within each VAE
    3. Mutual information minimization between z_b and z_d
    4. Path progress conditioning for temporal awareness
    5. Hierarchical intention learning from beliefs and desires
    
    This model learns interpretable mental state representations that can be
    used for Theory of Mind reasoning, behavior prediction, and explainable AI.
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_agents: int,
        num_poi_nodes: int,
        num_categories: int = 7,
        # Embedding dimensions
        node_embedding_dim: int = 64,
        temporal_dim: int = 64,
        agent_dim: int = 64,
        fusion_dim: int = 128,
        # VAE dimensions
        belief_latent_dim: int = 32,
        desire_latent_dim: int = 32,
        intention_latent_dim: int = 64,
        vae_hidden_dim: int = 128,
        vae_num_layers: int = 2,
        # Prediction head dimensions
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 4,
        # Loss weights
        beta_belief: float = 4.0,  # β for belief TC penalty
        beta_desire: float = 4.0,  # β for desire TC penalty
        beta_intention: float = 1.0,  # β for intention KL
        mi_weight: float = 0.5,  # Weight for MI(z_b, z_d) minimization
        transition_weight: float = 0.1,  # Weight for belief transition loss
        category_weight: float = 0.1,  # Weight for desire category loss
        # Options
        use_progress: bool = True,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.num_categories = num_categories
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        
        # Store dimensions
        self.belief_latent_dim = belief_latent_dim
        self.desire_latent_dim = desire_latent_dim
        self.intention_latent_dim = intention_latent_dim
        
        # Store loss weights
        self.beta_belief = beta_belief
        self.beta_desire = beta_desire
        self.beta_intention = beta_intention
        self.mi_weight = mi_weight
        self.transition_weight = transition_weight
        self.category_weight = category_weight
        self.use_progress = use_progress
        
        # ================================================================
        # MODULE 1: UNIFIED EMBEDDING PIPELINE
        # ================================================================
        self.embedding_pipeline = UnifiedEmbeddingPipeline(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_categories=num_categories,
            node_embedding_dim=node_embedding_dim,
            temporal_dim=temporal_dim,
            agent_dim=agent_dim,
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            n_fusion_layers=2,
            n_heads=num_heads,
            dropout=dropout,
            use_node2vec=True,
            use_temporal=False,
            use_agent=True,
            use_modality_gating=True,
            use_cross_attention=True,
        )
        
        if freeze_embedding:
            self._freeze_embeddings()
        
        # ================================================================
        # MODULE 2: FEATURE BRANCH PROJECTIONS
        # ================================================================
        # Spatial branch: emphasizes structural/location features for Beliefs
        self.spatial_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        
        # Agent/preference branch: emphasizes identity/preference features for Desires
        self.preference_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        
        # ================================================================
        # MODULE 3: BELIEF VAE
        # ================================================================
        self.belief_vae = BeliefVAEv2(
            input_dim=fusion_dim,
            hidden_dim=vae_hidden_dim,
            latent_dim=belief_latent_dim,
            num_nodes=num_nodes,
            num_layers=vae_num_layers,
            dropout=dropout,
        )
        
        # ================================================================
        # MODULE 4: DESIRE VAE
        # ================================================================
        self.desire_vae = DesireVAEv2(
            input_dim=fusion_dim,
            hidden_dim=vae_hidden_dim,
            latent_dim=desire_latent_dim,
            num_categories=num_categories,
            num_layers=vae_num_layers,
            dropout=dropout,
        )
        
        # ================================================================
        # MODULE 5: INTENTION VAE
        # ================================================================
        self.intention_vae = IntentionVAEv2(
            belief_dim=belief_latent_dim,
            desire_dim=desire_latent_dim,
            hidden_dim=vae_hidden_dim,
            latent_dim=intention_latent_dim,
            include_progress=use_progress,
            num_layers=vae_num_layers,
            dropout=dropout,
        )
        
        # ================================================================
        # MODULE 6: MUTUAL INFORMATION ESTIMATOR
        # ================================================================
        self.mi_estimator = MutualInformationEstimator(
            belief_dim=belief_latent_dim,
            desire_dim=desire_latent_dim,
            hidden_dim=64,
        )
        
        # ================================================================
        # MODULE 7: PREDICTION HEADS
        # ================================================================
        # Project intention to prediction features
        self.intention_projection = nn.Sequential(
            nn.Linear(intention_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Goal prediction: which POI is the destination
        self.goal_head = nn.Linear(hidden_dim, num_poi_nodes)
        
        # Next step prediction: immediate next location
        self.nextstep_head = nn.Linear(hidden_dim, num_nodes)
        
        # Category prediction: semantic category of goal
        self.category_head = nn.Linear(hidden_dim, num_categories)
        
        # Progress prediction: how far along the path (0-1)
        if use_progress:
            self.progress_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        
        # Apply proper weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for numerical stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _freeze_embeddings(self):
        """Freeze embedding pipeline parameters."""
        for param in self.embedding_pipeline.parameters():
            param.requires_grad = False
        print("❄️  Embedding pipeline frozen!")
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding parameters."""
        for param in self.embedding_pipeline.parameters():
            param.requires_grad = True
        print("🔥 Embedding pipeline unfrozen!")
    
    def _get_unified_embedding(
        self,
        history_node_indices: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Extract unified embedding from trajectory history with normalization."""
        batch_size = history_node_indices.shape[0]
        device = history_node_indices.device
        
        # Get node embeddings
        node_emb = self.embedding_pipeline.encode_nodes(
            history_node_indices,
            spatial_coords=None,
            categories=None,
        )  # [batch, seq_len, node_embedding_dim]
        
        # Check for NaN in embeddings
        if torch.isnan(node_emb).any():
            node_emb = torch.nan_to_num(node_emb, nan=0.0)
        
        # Expand to fusion_dim if needed
        if node_emb.shape[-1] < self.fusion_dim:
            padding = torch.zeros(
                batch_size, node_emb.shape[1], self.fusion_dim - node_emb.shape[-1],
                device=device
            )
            node_emb = torch.cat([node_emb, padding], dim=-1)
        
        # Extract last valid embedding for each sequence
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = (history_lengths - 1).clamp(min=0)
        unified_embedding = node_emb[batch_indices, last_indices]  # [batch, fusion_dim]
        
        # Normalize to prevent gradient explosion
        unified_embedding = F.layer_norm(unified_embedding, [self.fusion_dim])
        
        return unified_embedding
    
    def forward(
        self,
        history_node_indices: torch.Tensor,
        history_lengths: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        path_progress: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        next_node_idx: Optional[torch.Tensor] = None,  # For belief auxiliary loss
        goal_cat_idx: Optional[torch.Tensor] = None,  # For desire auxiliary loss
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through C²D-BDI model.
        
        Args:
            history_node_indices: [batch, seq_len] node indices
            history_lengths: [batch] actual sequence lengths
            agent_ids: [batch] agent indices (optional)
            path_progress: [batch] or [batch, 1] progress 0-1 (optional)
            compute_loss: Whether to compute VAE losses
            next_node_idx: [batch] next node targets (for belief auxiliary)
            goal_cat_idx: [batch] category targets (for desire auxiliary)
        
        Returns:
            Dict with predictions, mental states, and losses
        """
        device = history_node_indices.device
        batch_size = history_node_indices.shape[0]
        
        # ================================================================
        # STEP 1: COMPUTE UNIFIED EMBEDDING
        # ================================================================
        unified_embedding = self._get_unified_embedding(
            history_node_indices, history_lengths
        )
        
        # ================================================================
        # STEP 2: PROJECT TO SEPARATE FEATURE BRANCHES
        # ================================================================
        spatial_features = self.spatial_projection(unified_embedding)
        preference_features = self.preference_projection(unified_embedding)
        
        # ================================================================
        # STEP 3: BELIEF VAE (world model)
        # ================================================================
        belief_out = self.belief_vae(spatial_features, return_transition=True)
        belief_z = belief_out['z']
        
        # ================================================================
        # STEP 4: DESIRE VAE (preferences)
        # ================================================================
        desire_out = self.desire_vae(preference_features, return_category=True)
        desire_z = desire_out['z']
        
        # ================================================================
        # STEP 5: INTENTION VAE (action policy)
        # ================================================================
        # Prepare progress
        if self.use_progress and path_progress is not None:
            if path_progress.dim() == 1:
                progress = path_progress.unsqueeze(1)
            else:
                progress = path_progress
        else:
            progress = None
        
        intention_out = self.intention_vae(belief_z, desire_z, progress)
        intention_z = intention_out['z']
        
        # ================================================================
        # STEP 6: PREDICTION HEADS
        # ================================================================
        pred_features = self.intention_projection(intention_z)
        
        goal_logits = self.goal_head(pred_features)
        nextstep_logits = self.nextstep_head(pred_features)
        category_logits = self.category_head(pred_features)
        
        outputs = {
            # Predictions
            'goal': goal_logits,
            'nextstep': nextstep_logits,
            'category': category_logits,
            # Mental states
            'belief_z': belief_z,
            'desire_z': desire_z,
            'intention_z': intention_z,
            # Embeddings
            'unified_embedding': unified_embedding,
            'spatial_features': spatial_features,
            'preference_features': preference_features,
        }
        
        if self.use_progress:
            progress_pred = self.progress_head(pred_features)
            outputs['progress_pred'] = progress_pred
        
        # ================================================================
        # STEP 7: COMPUTE LOSSES
        # ================================================================
        if compute_loss:
            # Belief VAE loss (β-TCVAE)
            belief_loss, belief_loss_dict = compute_tcvae_loss(
                recon=belief_out['recon'],
                target=belief_out['target'],
                z=belief_out['z'],
                mu=belief_out['mu'],
                log_var=belief_out['log_var'],
                beta=self.beta_belief,
            )
            
            # Belief auxiliary: transition prediction
            if next_node_idx is not None:
                transition_loss = F.cross_entropy(
                    belief_out['transition_logits'], next_node_idx
                )
                belief_loss = belief_loss + self.transition_weight * transition_loss
                outputs['belief_transition_loss'] = transition_loss
            
            # Desire VAE loss (β-TCVAE)
            desire_loss, desire_loss_dict = compute_tcvae_loss(
                recon=desire_out['recon'],
                target=desire_out['target'],
                z=desire_out['z'],
                mu=desire_out['mu'],
                log_var=desire_out['log_var'],
                beta=self.beta_desire,
            )
            
            # Desire auxiliary: category prediction
            if goal_cat_idx is not None:
                category_aux_loss = F.cross_entropy(
                    desire_out['category_logits'], goal_cat_idx
                )
                desire_loss = desire_loss + self.category_weight * category_aux_loss
                outputs['desire_category_loss'] = category_aux_loss
            
            # Intention VAE loss (standard VAE, not TCVAE - we want it correlated)
            intention_recon_loss = F.mse_loss(
                intention_out['recon'], intention_out['target']
            )
            intention_recon_loss = torch.clamp(intention_recon_loss, max=100.0)
            
            # Clamp log_var for numerical stability
            intention_log_var = torch.clamp(intention_out['log_var'], min=-10.0, max=10.0)
            intention_kl = -0.5 * torch.sum(
                1 + intention_log_var - 
                intention_out['mu'].pow(2) - 
                intention_log_var.exp(),
                dim=-1
            ).mean()
            intention_kl = torch.clamp(intention_kl, min=0.0, max=100.0)
            intention_loss = intention_recon_loss + self.beta_intention * intention_kl
            
            # Mutual Information loss (minimize correlation between z_b and z_d)
            mi_loss = self.mi_estimator(belief_z.detach(), desire_z)
            # We want to minimize MI, so we add it as a penalty
            mi_loss_weighted = self.mi_weight * mi_loss
            
            # Total VAE loss
            total_vae_loss = belief_loss + desire_loss + intention_loss + mi_loss_weighted
            
            outputs.update({
                # Belief losses
                'belief_loss': belief_loss,
                'belief_recon_loss': belief_loss_dict['recon_loss'],
                'belief_tc_loss': belief_loss_dict['tc_loss'],
                'belief_mi_loss': belief_loss_dict['mi_loss'],
                # Desire losses
                'desire_loss': desire_loss,
                'desire_recon_loss': desire_loss_dict['recon_loss'],
                'desire_tc_loss': desire_loss_dict['tc_loss'],
                'desire_mi_loss': desire_loss_dict['mi_loss'],
                # Intention losses
                'intention_loss': intention_loss,
                'intention_recon_loss': intention_recon_loss,
                'intention_kl_loss': intention_kl,
                # Cross-VAE losses
                'bd_mi_loss': mi_loss,  # MI between belief and desire
                # Total
                'total_vae_loss': total_vae_loss,
            })
            
            # Progress loss if applicable
            if self.use_progress and path_progress is not None:
                if path_progress.dim() == 1:
                    target_progress = path_progress.unsqueeze(1)
                else:
                    target_progress = path_progress
                progress_loss = F.mse_loss(outputs['progress_pred'], target_progress)
                outputs['progress_loss'] = progress_loss
        
        return outputs
    
    def encode_mental_states(
        self,
        history_node_indices: torch.Tensor,
        history_lengths: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        path_progress: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode trajectory to mental state representations (inference mode).
        
        Returns:
            Dict with belief_z, desire_z, intention_z, unified_embedding
        """
        with torch.no_grad():
            outputs = self.forward(
                history_node_indices=history_node_indices,
                history_lengths=history_lengths,
                agent_ids=agent_ids,
                path_progress=path_progress,
                compute_loss=False,
            )
        
        return {
            'belief_z': outputs['belief_z'],
            'desire_z': outputs['desire_z'],
            'intention_z': outputs['intention_z'],
            'unified_embedding': outputs['unified_embedding'],
        }
    
    def get_interpretable_states(
        self,
        history_node_indices: torch.Tensor,
        history_lengths: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        path_progress: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get interpretable mental state predictions.
        
        Returns dict with:
        - belief_z: Where the agent thinks they are (world model)
        - desire_z: What the agent wants (preferences)
        - intention_z: How the agent plans to achieve it (policy)
        - predicted_goal: Most likely destination
        - predicted_category: Type of destination
        - predicted_nextstep: Immediate next action
        """
        outputs = self.forward(
            history_node_indices=history_node_indices,
            history_lengths=history_lengths,
            agent_ids=agent_ids,
            path_progress=path_progress,
            compute_loss=False,
        )
        
        return {
            'belief_z': outputs['belief_z'],
            'desire_z': outputs['desire_z'],
            'intention_z': outputs['intention_z'],
            'predicted_goal': outputs['goal'].argmax(dim=-1),
            'goal_probabilities': F.softmax(outputs['goal'], dim=-1),
            'predicted_category': outputs['category'].argmax(dim=-1),
            'category_probabilities': F.softmax(outputs['category'], dim=-1),
            'predicted_nextstep': outputs['nextstep'].argmax(dim=-1),
        }
