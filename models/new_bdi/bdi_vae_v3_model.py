"""
SEQUENTIAL CONDITIONAL BDI-VAE (SC-BDI) - VERSION 3

Implements the corrected architecture based on rigorous critique:

KEY IMPROVEMENTS OVER V2:
1. CONDITIONAL PRIOR for Intention: p(z_i | z_b, z_d) instead of N(0,I)
2. CONTRASTIVE LOSS (InfoNCE) between z_d and goal - forces desires to predict goals
3. KL ANNEALING with free-bits to prevent posterior collapse
4. GOAL-SUPERVISED DESIRE: z_d directly predicts goal (not just category)
5. STRONGER CAUSAL STRUCTURE: z_i = f(z_b, z_d) with learned conditional prior

The core insight from the critique:
- Desire (z_d) should DIRECTLY predict the final goal via contrastive learning
- Intention (z_i) should have a CONDITIONAL prior p(z_i | z_b, z_d), not N(0,I)
- KL annealing + free-bits prevents posterior collapse

ARCHITECTURE:
    Observations → Encoder → Unified Embedding
                                    ↓
            ┌───────────────────────┴───────────────────────┐
            ↓                                               ↓
    [Spatial Branch]                               [Preference Branch]
            ↓                                               ↓
    ┌───────────────┐                             ┌───────────────┐
    │   BELIEF VAE  │                             │  DESIRE VAE   │
    │   p(z_b)      │                             │   p(z_d)      │
    │   World Model │                             │   + InfoNCE   │ ← Contrastive to Goal!
    └───────┬───────┘                             └───────┬───────┘
            │                                             │
            └──────────────────┬──────────────────────────┘
                               ↓
                    ┌─────────────────────┐
                    │    INTENTION VAE    │
                    │ p(z_i | z_b, z_d)   │ ← CONDITIONAL PRIOR!
                    │    Action Policy    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   PREDICTION HEADS  │
                    │   Goal, Category    │
                    └─────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math

from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline


# =============================================================================
# GAUSSIAN ENCODER/DECODER (same as V2 but cleaner)
# =============================================================================

class GaussianEncoder(nn.Module):
    """Encoder outputting diagonal Gaussian parameters."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize log-variance head for moderate initial variance
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=2.0)
        return mu, logvar


class GaussianDecoder(nn.Module):
    """Decoder from latent to output space."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick."""
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# =============================================================================
# CONDITIONAL PRIOR NETWORK (KEY IMPROVEMENT!)
# =============================================================================

class ConditionalPriorNetwork(nn.Module):
    """
    Learns p(z_i | z_b, z_d) - the conditional prior for intentions.
    
    This is THE KEY MISSING PIECE from V2!
    Instead of using N(0, I) as prior for z_i, we learn:
        p(z_i | z_b, z_d) = N(mu_prior(z_b, z_d), sigma_prior(z_b, z_d))
    
    This properly encodes the causal structure: intentions are CAUSED by
    beliefs and desires, so the prior should depend on them.
    """
    
    def __init__(
        self,
        belief_dim: int,
        desire_dim: int,
        intention_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        input_dim = belief_dim + desire_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim, intention_dim)
        self.fc_logvar = nn.Linear(hidden_dim, intention_dim)
        
        # Initialize to output near-standard normal initially
        nn.init.zeros_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)
    
    def forward(
        self, 
        belief_z: torch.Tensor, 
        desire_z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional prior parameters p(z_i | z_b, z_d).
        
        Returns:
            prior_mu: [batch, intention_dim]
            prior_logvar: [batch, intention_dim]
        """
        x = torch.cat([belief_z, desire_z], dim=-1)
        h = self.net(x)
        prior_mu = self.fc_mu(h)
        prior_logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=2.0)
        return prior_mu, prior_logvar


# =============================================================================
# INFONCE CONTRASTIVE LOSS (KEY IMPROVEMENT!)
# =============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss between desire latent z_d and goal embeddings.
    
    This forces z_d to contain information predictive of the FINAL GOAL.
    Without this, desires have no pressure to relate to goals!
    
    L_InfoNCE = -log(exp(sim(z_d, g_pos)/τ) / Σ exp(sim(z_d, g_neg)/τ))
    """
    
    def __init__(
        self,
        desire_dim: int,
        goal_embedding_dim: int,
        hidden_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Project both to shared space for comparison
        self.desire_proj = nn.Sequential(
            nn.Linear(desire_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.goal_proj = nn.Sequential(
            nn.Linear(goal_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        desire_z: torch.Tensor,  # [batch, desire_dim]
        goal_embeddings: torch.Tensor,  # [batch, goal_embedding_dim]
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Positive pairs: (z_d[i], goal[i]) from same sample
        Negative pairs: (z_d[i], goal[j]) where j != i
        """
        batch_size = desire_z.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=desire_z.device)
        
        # Project to shared space
        z_proj = F.normalize(self.desire_proj(desire_z), dim=-1)  # [B, H]
        g_proj = F.normalize(self.goal_proj(goal_embeddings), dim=-1)  # [B, H]
        
        # Compute similarity matrix [B, B]
        sim_matrix = torch.matmul(z_proj, g_proj.T) / self.temperature
        
        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=desire_z.device)
        
        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


# =============================================================================
# FREE-BITS KL LOSS (prevents posterior collapse)
# =============================================================================

def kl_divergence_free_bits(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.5,
) -> torch.Tensor:
    """
    KL divergence with free-bits: each latent dimension must use at least
    'free_bits' nats before contributing to the loss.
    
    This prevents posterior collapse where VAE ignores latent codes.
    """
    # Per-dimension KL
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
    
    # Apply free-bits: clamp minimum to free_bits nats per dimension
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    
    # Sum over dimensions, mean over batch
    kl_loss = kl_per_dim.sum(dim=-1).mean()
    
    return kl_loss


def kl_divergence_conditional(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """
    KL divergence between two Gaussians: KL(q || p).
    
    Used for intention VAE where prior is conditional: p(z_i | z_b, z_d).
    
    KL(N(μ_q, σ_q²) || N(μ_p, σ_p²)) = 
        log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²) / (2σ_p²) - 1/2
    """
    var_q = q_logvar.exp()
    var_p = p_logvar.exp().clamp(min=1e-6)
    
    kl_per_dim = (
        0.5 * p_logvar - 0.5 * q_logvar +
        (var_q + (q_mu - p_mu).pow(2)) / (2 * var_p) -
        0.5
    )
    
    # Free-bits per dimension
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    
    return kl_per_dim.sum(dim=-1).mean()


# =============================================================================
# BELIEF VAE
# =============================================================================

class BeliefVAE(nn.Module):
    """
    Belief VAE: Models the agent's understanding of the world.
    
    Reconstructs spatial features and predicts possible transitions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_nodes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = GaussianEncoder(input_dim, hidden_dim, latent_dim, dropout=dropout)
        self.decoder = GaussianDecoder(latent_dim, hidden_dim, input_dim, dropout=dropout)
        self.transition_head = nn.Linear(latent_dim, num_nodes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        transition_logits = self.transition_head(z)
        
        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'recon': recon,
            'target': x,
            'transition_logits': transition_logits,
        }


# =============================================================================
# DESIRE VAE (with goal prediction head!)
# =============================================================================

class DesireVAE(nn.Module):
    """
    Desire VAE: Models agent preferences and goals.
    
    KEY ADDITION: Direct goal prediction head!
    z_d should directly predict the goal POI, not just category.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_categories: int,
        num_goals: int,  # NEW: number of POI goals
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = GaussianEncoder(input_dim, hidden_dim, latent_dim, dropout=dropout)
        self.decoder = GaussianDecoder(latent_dim, hidden_dim, input_dim, dropout=dropout)
        
        # Category prediction (what TYPE of goal)
        self.category_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_categories),
        )
        
        # DIRECT goal prediction (which specific POI)
        # This forces z_d to contain goal-relevant information!
        self.goal_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_goals),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        category_logits = self.category_head(z)
        goal_logits = self.goal_head(z)  # NEW!
        
        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'recon': recon,
            'target': x,
            'category_logits': category_logits,
            'goal_logits': goal_logits,  # NEW!
        }


# =============================================================================
# INTENTION VAE (with CONDITIONAL prior!)
# =============================================================================

class IntentionVAE(nn.Module):
    """
    Intention VAE: Models action policy conditioned on beliefs and desires.
    
    KEY IMPROVEMENT: Uses CONDITIONAL prior p(z_i | z_b, z_d)!
    This is crucial for proper causal structure.
    """
    
    def __init__(
        self,
        belief_dim: int,
        desire_dim: int,
        hidden_dim: int,
        latent_dim: int,
        include_progress: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.include_progress = include_progress
        input_dim = belief_dim + desire_dim + (1 if include_progress else 0)
        
        # Encoder: q(z_i | z_b, z_d, progress)
        self.encoder = GaussianEncoder(input_dim, hidden_dim, latent_dim, dropout=dropout)
        
        # Decoder: reconstructs [z_b, z_d, progress]
        self.decoder = GaussianDecoder(latent_dim, hidden_dim, input_dim, dropout=dropout)
        
        # CONDITIONAL PRIOR NETWORK: p(z_i | z_b, z_d)
        self.prior_net = ConditionalPriorNetwork(
            belief_dim=belief_dim,
            desire_dim=desire_dim,
            intention_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
    
    def forward(
        self,
        belief_z: torch.Tensor,
        desire_z: torch.Tensor,
        progress: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Build input
        if self.include_progress and progress is not None:
            if progress.dim() == 1:
                progress = progress.unsqueeze(1)
            x = torch.cat([belief_z, desire_z, progress], dim=-1)
        else:
            x = torch.cat([belief_z, desire_z], dim=-1)
        
        # Posterior: q(z_i | x)
        q_mu, q_logvar = self.encoder(x)
        z = reparameterize(q_mu, q_logvar)
        
        # Conditional prior: p(z_i | z_b, z_d)
        p_mu, p_logvar = self.prior_net(belief_z, desire_z)
        
        # Reconstruction
        recon = self.decoder(z)
        
        return {
            'z': z,
            'q_mu': q_mu,
            'q_logvar': q_logvar,
            'p_mu': p_mu,  # Conditional prior mean
            'p_logvar': p_logvar,  # Conditional prior variance
            'recon': recon,
            'target': x,
        }


# =============================================================================
# MUTUAL INFORMATION ESTIMATOR (same as V2)
# =============================================================================

class MutualInformationEstimator(nn.Module):
    """MINE-based MI estimator for disentanglement."""
    
    def __init__(self, belief_dim: int, desire_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(belief_dim + desire_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, belief_z: torch.Tensor, desire_z: torch.Tensor) -> torch.Tensor:
        batch_size = belief_z.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=belief_z.device)
        
        joint = torch.cat([belief_z, desire_z], dim=-1)
        shuffle_idx = torch.randperm(batch_size, device=desire_z.device)
        marginal = torch.cat([belief_z, desire_z[shuffle_idx]], dim=-1)
        
        joint_scores = torch.clamp(self.discriminator(joint), -10, 10)
        marginal_scores = torch.clamp(self.discriminator(marginal), -10, 10)
        
        mi = torch.mean(joint_scores) - (
            torch.logsumexp(marginal_scores.squeeze(-1), dim=0) - math.log(batch_size)
        )
        
        return torch.clamp(mi, -10, 10)


# =============================================================================
# MAIN MODEL: SC-BDI-VAE V3
# =============================================================================

class SequentialConditionalBDIVAE(nn.Module):
    """
    Sequential Conditional BDI-VAE (SC-BDI) V3
    
    Key improvements over V2:
    1. Conditional prior for intention: p(z_i | z_b, z_d)
    2. InfoNCE contrastive loss for desire-to-goal alignment
    3. Direct goal prediction from z_d
    4. Free-bits KL to prevent posterior collapse
    5. KL annealing support
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_agents: int,
        num_poi_nodes: int,
        num_categories: int = 7,
        # Embedding dimensions
        node_embedding_dim: int = 64,
        fusion_dim: int = 128,
        # VAE dimensions
        belief_latent_dim: int = 32,
        desire_latent_dim: int = 16,  # Smaller! Forces abstraction
        intention_latent_dim: int = 32,
        vae_hidden_dim: int = 128,
        # Prediction
        hidden_dim: int = 256,
        dropout: float = 0.1,
        # Loss weights
        beta_belief: float = 1.0,
        beta_desire: float = 1.0,
        beta_intention: float = 1.0,
        mi_weight: float = 0.1,
        infonce_weight: float = 1.0,  # NEW!
        desire_goal_weight: float = 0.5,  # NEW!
        transition_weight: float = 0.1,
        category_weight: float = 0.1,
        # Free-bits
        free_bits: float = 0.5,  # NEW!
        # Options
        use_progress: bool = False,  # Disabled to test training without path progress
        use_temporal: bool = False,  # Whether to include temporal encoding
        infonce_temperature: float = 0.1,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.num_categories = num_categories
        self.fusion_dim = fusion_dim
        
        # Store dimensions
        self.belief_latent_dim = belief_latent_dim
        self.desire_latent_dim = desire_latent_dim
        self.intention_latent_dim = intention_latent_dim
        
        # Store weights
        self.beta_belief = beta_belief
        self.beta_desire = beta_desire
        self.beta_intention = beta_intention
        self.mi_weight = mi_weight
        self.infonce_weight = infonce_weight
        self.desire_goal_weight = desire_goal_weight
        self.transition_weight = transition_weight
        self.category_weight = category_weight
        self.free_bits = free_bits
        self.use_progress = use_progress
        
        # KL annealing (controlled externally)
        self.kl_weight = 1.0  # Will be annealed during training
        
        # ================================================================
        # EMBEDDING PIPELINE
        # ================================================================
        self.embedding_pipeline = UnifiedEmbeddingPipeline(
            num_nodes=num_nodes,
            num_agents=num_agents,
            num_categories=num_categories,
            node_embedding_dim=node_embedding_dim,
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            use_node2vec=True,
            use_temporal=True,   # ENABLED: full multi-modal fusion
            use_agent=True,
        )
        
        # ================================================================
        # FEATURE PROJECTIONS
        # ================================================================
        self.spatial_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        
        self.preference_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        
        # ================================================================
        # VAEs
        # ================================================================
        self.belief_vae = BeliefVAE(
            input_dim=fusion_dim,
            hidden_dim=vae_hidden_dim,
            latent_dim=belief_latent_dim,
            num_nodes=num_nodes,
            dropout=dropout,
        )
        
        self.desire_vae = DesireVAE(
            input_dim=fusion_dim,
            hidden_dim=vae_hidden_dim,
            latent_dim=desire_latent_dim,
            num_categories=num_categories,
            num_goals=num_poi_nodes,  # NEW: direct goal prediction
            dropout=dropout,
        )
        
        self.intention_vae = IntentionVAE(
            belief_dim=belief_latent_dim,
            desire_dim=desire_latent_dim,
            hidden_dim=vae_hidden_dim,
            latent_dim=intention_latent_dim,
            include_progress=use_progress,
            dropout=dropout,
        )
        
        # ================================================================
        # DISENTANGLEMENT
        # ================================================================
        self.mi_estimator = MutualInformationEstimator(
            belief_dim=belief_latent_dim,
            desire_dim=desire_latent_dim,
        )
        
        # ================================================================
        # CONTRASTIVE LOSS (InfoNCE)
        # ================================================================
        # Goal embedding: learnable embeddings for each POI
        self.goal_embeddings = nn.Embedding(num_poi_nodes, fusion_dim)
        
        self.infonce = InfoNCELoss(
            desire_dim=desire_latent_dim,
            goal_embedding_dim=fusion_dim,
            hidden_dim=vae_hidden_dim,
            temperature=infonce_temperature,
        )
        
        # ================================================================
        # PREDICTION HEADS
        # ================================================================
        self.intention_projection = nn.Sequential(
            nn.Linear(intention_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Final prediction: from intention latent
        self.goal_head = nn.Linear(hidden_dim, num_poi_nodes)
        self.nextstep_head = nn.Linear(hidden_dim, num_nodes)
        self.category_head = nn.Linear(hidden_dim, num_categories)
        
        if use_progress:
            self.progress_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def set_kl_weight(self, weight: float):
        """Set KL weight for annealing."""
        self.kl_weight = weight
    
    def _get_unified_embedding(
        self,
        history_node_indices: torch.Tensor,
        history_lengths: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        hours: Optional[torch.Tensor] = None,
        days: Optional[torch.Tensor] = None,
        deltas: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute a unified embedding via the full multi-modal pipeline.

        When temporal / agent information is provided, the pipeline fuses
        node2vec, temporal, and agent modalities.  When they are absent it
        falls back to node embeddings only (backward-compatible).
        """
        batch_size = history_node_indices.shape[0]
        seq_len = history_node_indices.shape[1]
        device = history_node_indices.device

        has_temporal = (hours is not None) and self.embedding_pipeline.use_temporal
        has_agent = (agent_ids is not None) and self.embedding_pipeline.use_agent

        if has_temporal or has_agent:
            # ---------- Full multi-modal fusion path ----------
            # Build a padding mask: 1 for valid positions, 0 for padding
            positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
            mask = (positions < history_lengths.unsqueeze(1)).float()       # (B, S)

            # Expand scalar hour / day to sequence-level for the temporal encoder
            if hours is not None and hours.dim() == 1:
                hours_seq = hours.unsqueeze(1).expand(-1, seq_len)  # (B, S)
            else:
                hours_seq = hours if hours is not None else torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

            if days is not None and days.dim() == 1:
                days_seq = days.unsqueeze(1).expand(-1, seq_len)
            else:
                days_seq = days if days is not None else torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

            # Deltas / velocities are already (B, S)
            if deltas is None:
                deltas = torch.zeros(batch_size, seq_len, device=device)
            if velocities is None:
                velocities = torch.zeros(batch_size, seq_len, device=device)

            # Forward through full pipeline → (B, S, fusion_dim) per-node
            fused_per_node = self.embedding_pipeline(
                node_ids=history_node_indices,
                agent_ids=agent_ids,
                hours=hours_seq,
                days=days_seq,
                deltas=deltas,
                velocities=velocities,
                mask=mask,
                return_per_node=True,
            )
            # If the pipeline returns a tuple (emb, components), take the emb
            if isinstance(fused_per_node, tuple):
                fused_per_node = fused_per_node[0]

            # Pick last-valid-step embedding per sequence
            batch_indices = torch.arange(batch_size, device=device)
            last_indices = (history_lengths - 1).clamp(min=0)
            unified = fused_per_node[batch_indices, last_indices]  # (B, fusion_dim)
        else:
            # ---------- Fallback: node embeddings only ----------
            node_emb = self.embedding_pipeline.encode_nodes(
                history_node_indices, spatial_coords=None, categories=None
            )

            if torch.isnan(node_emb).any():
                node_emb = torch.nan_to_num(node_emb, nan=0.0)

            if node_emb.shape[-1] < self.fusion_dim:
                padding = torch.zeros(
                    batch_size, node_emb.shape[1], self.fusion_dim - node_emb.shape[-1],
                    device=device
                )
                node_emb = torch.cat([node_emb, padding], dim=-1)

            batch_indices = torch.arange(batch_size, device=device)
            last_indices = (history_lengths - 1).clamp(min=0)
            unified = node_emb[batch_indices, last_indices]

        return F.layer_norm(unified, [self.fusion_dim])
    
    def forward(
        self,
        history_node_indices: torch.Tensor,
        history_lengths: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        path_progress: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        next_node_idx: Optional[torch.Tensor] = None,
        goal_idx: Optional[torch.Tensor] = None,  # IMPORTANT: need goal for InfoNCE!
        goal_cat_idx: Optional[torch.Tensor] = None,
        # --- Temporal features for full unified pipeline ---
        hours: Optional[torch.Tensor] = None,
        days: Optional[torch.Tensor] = None,
        deltas: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with all losses."""
        
        device = history_node_indices.device
        batch_size = history_node_indices.shape[0]
        
        # ================================================================
        # EMBEDDING (full multi-modal when temporal features are provided)
        # ================================================================
        unified_embedding = self._get_unified_embedding(
            history_node_indices, history_lengths,
            agent_ids=agent_ids,
            hours=hours,
            days=days,
            deltas=deltas,
            velocities=velocities,
        )
        
        spatial_features = self.spatial_projection(unified_embedding)
        preference_features = self.preference_projection(unified_embedding)
        
        # ================================================================
        # BELIEF VAE
        # ================================================================
        belief_out = self.belief_vae(spatial_features)
        belief_z = belief_out['z']
        
        # ================================================================
        # DESIRE VAE
        # ================================================================
        desire_out = self.desire_vae(preference_features)
        desire_z = desire_out['z']
        
        # ================================================================
        # INTENTION VAE (with conditional prior!)
        # ================================================================
        if self.use_progress and path_progress is not None:
            progress = path_progress.unsqueeze(1) if path_progress.dim() == 1 else path_progress
        else:
            progress = None
        
        intention_out = self.intention_vae(belief_z, desire_z, progress)
        intention_z = intention_out['z']
        
        # ================================================================
        # PREDICTIONS
        # ================================================================
        pred_features = self.intention_projection(intention_z)
        
        goal_logits = self.goal_head(pred_features)
        nextstep_logits = self.nextstep_head(pred_features)
        category_logits = self.category_head(pred_features)
        
        outputs = {
            'goal': goal_logits,
            'nextstep': nextstep_logits,
            'category': category_logits,
            'belief_z': belief_z,
            'desire_z': desire_z,
            'intention_z': intention_z,
            'desire_goal_logits': desire_out['goal_logits'],  # NEW!
        }
        
        if self.use_progress:
            outputs['progress_pred'] = self.progress_head(pred_features)
        
        # ================================================================
        # LOSSES
        # ================================================================
        if compute_loss:
            # --- Belief VAE loss ---
            belief_recon_loss = F.mse_loss(belief_out['recon'], belief_out['target'])
            belief_kl = kl_divergence_free_bits(
                belief_out['mu'], belief_out['logvar'], self.free_bits
            )
            belief_loss = belief_recon_loss + self.kl_weight * self.beta_belief * belief_kl
            
            if next_node_idx is not None:
                transition_loss = F.cross_entropy(
                    belief_out['transition_logits'], next_node_idx
                )
                belief_loss = belief_loss + self.transition_weight * transition_loss
                outputs['transition_loss'] = transition_loss
            
            # --- Desire VAE loss ---
            desire_recon_loss = F.mse_loss(desire_out['recon'], desire_out['target'])
            desire_kl = kl_divergence_free_bits(
                desire_out['mu'], desire_out['logvar'], self.free_bits
            )
            desire_loss = desire_recon_loss + self.kl_weight * self.beta_desire * desire_kl
            
            if goal_cat_idx is not None:
                category_loss = F.cross_entropy(
                    desire_out['category_logits'], goal_cat_idx
                )
                desire_loss = desire_loss + self.category_weight * category_loss
                outputs['desire_category_loss'] = category_loss
            
            # NEW: Direct goal prediction from desire!
            if goal_idx is not None:
                desire_goal_loss = F.cross_entropy(
                    desire_out['goal_logits'], goal_idx
                )
                desire_loss = desire_loss + self.desire_goal_weight * desire_goal_loss
                outputs['desire_goal_loss'] = desire_goal_loss
            
            # --- Intention VAE loss (with CONDITIONAL prior!) ---
            intention_recon_loss = F.mse_loss(intention_out['recon'], intention_out['target'])
            
            # KL to CONDITIONAL prior, not N(0,I)!
            intention_kl = kl_divergence_conditional(
                q_mu=intention_out['q_mu'],
                q_logvar=intention_out['q_logvar'],
                p_mu=intention_out['p_mu'],
                p_logvar=intention_out['p_logvar'],
                free_bits=self.free_bits,
            )
            intention_loss = intention_recon_loss + self.kl_weight * self.beta_intention * intention_kl
            
            # --- MI minimization (disentangle belief and desire) ---
            mi_loss = self.mi_estimator(belief_z.detach(), desire_z)
            
            # --- InfoNCE contrastive loss (desire ↔ goal) ---
            infonce_loss = torch.tensor(0.0, device=device)
            if goal_idx is not None:
                goal_emb = self.goal_embeddings(goal_idx)  # [B, fusion_dim]
                infonce_loss = self.infonce(desire_z, goal_emb)
            
            # --- Total VAE loss ---
            total_vae_loss = (
                belief_loss + 
                desire_loss + 
                intention_loss + 
                self.mi_weight * mi_loss +
                self.infonce_weight * infonce_loss
            )
            
            outputs.update({
                'belief_loss': belief_loss,
                'belief_recon_loss': belief_recon_loss,
                'belief_kl': belief_kl,
                'desire_loss': desire_loss,
                'desire_recon_loss': desire_recon_loss,
                'desire_kl': desire_kl,
                'intention_loss': intention_loss,
                'intention_recon_loss': intention_recon_loss,
                'intention_kl': intention_kl,
                'mi_loss': mi_loss,
                'infonce_loss': infonce_loss,
                'total_vae_loss': total_vae_loss,
            })
            
            if self.use_progress and path_progress is not None:
                target = path_progress.unsqueeze(1) if path_progress.dim() == 1 else path_progress
                outputs['progress_loss'] = F.mse_loss(outputs['progress_pred'], target)
        
        return outputs
    
    def encode_mental_states(
        self,
        history_node_indices: torch.Tensor,
        history_lengths: torch.Tensor,
        path_progress: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference-only encoding."""
        with torch.no_grad():
            outputs = self.forward(
                history_node_indices=history_node_indices,
                history_lengths=history_lengths,
                path_progress=path_progress,
                compute_loss=False,
            )
        
        return {
            'belief_z': outputs['belief_z'],
            'desire_z': outputs['desire_z'],
            'intention_z': outputs['intention_z'],
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_sc_bdi_vae_v3(
    num_nodes: int,
    num_agents: int,
    num_poi_nodes: int,
    num_categories: int = 7,
    **kwargs,
) -> SequentialConditionalBDIVAE:
    """Create SC-BDI-VAE V3 model."""
    return SequentialConditionalBDIVAE(
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_poi_nodes=num_poi_nodes,
        num_categories=num_categories,
        **kwargs,
    )
