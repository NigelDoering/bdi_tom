"""
HIERARCHICAL BDI VARIATIONAL AUTOENCODER MODEL

This module implements the main BDI (Belief-Desire-Intention) VAE model
for learning mental state representations without labels.

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                   UNIFIED EMBEDDING PIPELINE                     │
│     (Graph + Trajectory + Agent + Temporal Context)             │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
  ┌─────────┐           ┌──────────┐
  │ BELIEF  │           │ DESIRE   │
  │   VAE   │           │   VAE    │
  └────┬────┘           └────┬─────┘
       │                     │
       │ belief_z            │ desire_z
       │                     │
       └──────────┬──────────┘
                  │
                  ├──── unified_embedding (skip connection)
                  │
                  ▼
          ┌───────────────┐
          │  INTENTION    │
          │     VAE       │
          └───────┬───────┘
                  │
                  │ intention_z
                  │
                  ▼
          ┌───────────────┐
          │  PREDICTION   │
          │    HEADS      │
          └───────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
     GOAL    NEXT STEP  CATEGORY


MENTAL STATE REPRESENTATIONS:
1. Belief VAE: Learns world model (spatial beliefs about environment)
   - Input: Unified embedding
   - Latent: Belief representation (where things are)
   - Reconstruction: Unified embedding

2. Desire VAE: Learns preferences (what agents want)
   - Input: Unified embedding (agent preferences encoded)
   - Latent: Desire representation (goal values)
   - Reconstruction: Unified embedding

3. Intention VAE: Learns goal-directed plans (how to achieve desires given beliefs)
   - Input: Concatenate [belief_z, desire_z, unified_embedding]
   - Latent: Intention representation (action policies)
   - Reconstruction: Concatenated input

TRAINING:
- Per-node training (like LSTM model)
- Multi-task loss = VAE losses + Prediction losses
  * Belief VAE loss: recon + β₁ * KL
  * Desire VAE loss: recon + β₂ * KL
  * Intention VAE loss: recon + β₃ * KL
  * Goal prediction loss
  * Next step prediction loss
  * Category prediction loss

UNSUPERVISED LEARNING:
- No labels needed for mental states
- VAE reconstruction forces disentangled representations
- Causal structure (Beliefs + Desires → Intentions) enforces BDI hierarchy
- Prediction tasks provide supervisory signal for intention quality
"""

import os
from typing import Dict, Tuple

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn

from models.en_encoders.unified_embedding_pipeline import UnifiedEmbeddingPipeline
from models.vae_bdi_simple.vae_components import (
    BeliefVAE,
    DesireVAE,
    IntentionVAE,
    vae_loss,
)


class BDIVAEPredictor(nn.Module):
    """
    Hierarchical BDI VAE Model with Unified Embedding Pipeline.
    
    DESIGN PHILOSOPHY:
    - Module 1: Unified Embedding Pipeline (shared representations)
    - Module 2: Parallel Belief + Desire VAEs (independent mental states)
    - Module 3: Intention VAE (hierarchical integration)
    - Module 4: Prediction Heads (supervised signal)
    
    INPUT: Trajectory history (node indices, lengths, agent_id)
    OUTPUT: Predictions + Mental state representations + VAE losses
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_agents: int,
        num_poi_nodes: int,
        num_categories: int = 7,
        # Embedding params
        node_embedding_dim: int = 64,
        temporal_dim: int = 64,
        agent_dim: int = 64,
        fusion_dim: int = 128,
        # VAE params
        belief_latent_dim: int = 32,
        desire_latent_dim: int = 32,
        intention_latent_dim: int = 64,
        vae_hidden_dim: int = 128,
        vae_num_layers: int = 2,
        # Prediction head params
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 4,
        # Loss weights (β-VAE)
        beta_belief: float = 1.0,
        beta_desire: float = 1.0,
        beta_intention: float = 1.0,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_poi_nodes = num_poi_nodes
        self.num_categories = num_categories
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.belief_latent_dim = belief_latent_dim
        self.desire_latent_dim = desire_latent_dim
        self.intention_latent_dim = intention_latent_dim
        
        # β-VAE weights
        self.beta_belief = beta_belief
        self.beta_desire = beta_desire
        self.beta_intention = beta_intention
        
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
        # MODULE 2: PARALLEL BELIEF + DESIRE VAEs
        # ================================================================
        
        # Belief VAE: Learns world model from unified embedding
        self.belief_vae = BeliefVAE(
            input_dim=fusion_dim,
            hidden_dim=vae_hidden_dim,
            latent_dim=belief_latent_dim,
            num_layers=vae_num_layers,
            dropout=dropout,
        )
        
        # Desire VAE: Learns preferences from unified embedding
        self.desire_vae = DesireVAE(
            input_dim=fusion_dim,
            hidden_dim=vae_hidden_dim,
            latent_dim=desire_latent_dim,
            num_layers=vae_num_layers,
            dropout=dropout,
        )
        
        # ================================================================
        # MODULE 3: HIERARCHICAL INTENTION VAE
        # ================================================================
        
        # Intention VAE: Integrates belief + desire + embedding
        self.intention_vae = IntentionVAE(
            belief_latent_dim=belief_latent_dim,
            desire_latent_dim=desire_latent_dim,
            embedding_dim=fusion_dim,  # Skip connection
            hidden_dim=vae_hidden_dim,
            latent_dim=intention_latent_dim,
            num_layers=vae_num_layers,
            dropout=dropout,
        )
        
        # ================================================================
        # MODULE 4: PREDICTION HEADS
        # ================================================================
        
        # Feature fusion: intention latent → prediction features
        self.feature_projection = nn.Sequential(
            nn.Linear(intention_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Prediction heads
        self.goal_head = nn.Linear(hidden_dim, num_poi_nodes)
        self.nextstep_head = nn.Linear(hidden_dim, num_nodes)
        self.category_head = nn.Linear(hidden_dim, num_categories)
    
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
    
    def forward(
        self,
        history_node_indices: torch.Tensor,  # [batch, seq_len]
        history_lengths: torch.Tensor,        # [batch]
        agent_ids: torch.Tensor = None,       # [batch] agent indices
        compute_loss: bool = True,
        free_bits: float = 2.0,               # Free bits for KL collapse prevention
        kl_annealing_factor: float = 1.0,     # KL annealing schedule (0→1)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical BDI VAE.
        
        Args:
            history_node_indices: [batch, seq_len] node indices
            history_lengths: [batch] actual sequence lengths
            agent_ids: [batch] agent indices (optional, for agent encoder)
            compute_loss: Whether to compute VAE losses
        
        Returns:
            Dict containing:
                Predictions:
                - 'goal': [batch, num_poi_nodes] goal logits
                - 'nextstep': [batch, num_nodes] next step logits
                - 'category': [batch, num_categories] category logits
                
                Mental State Representations:
                - 'belief_z': [batch, belief_latent_dim]
                - 'desire_z': [batch, desire_latent_dim]
                - 'intention_z': [batch, intention_latent_dim]
                
                VAE Components (if compute_loss=True):
                - 'belief_loss', 'belief_recon_loss', 'belief_kl_loss'
                - 'desire_loss', 'desire_recon_loss', 'desire_kl_loss'
                - 'intention_loss', 'intention_recon_loss', 'intention_kl_loss'
                - 'total_vae_loss'
                
                Embeddings:
                - 'unified_embedding': [batch, fusion_dim]
        """
        batch_size = history_node_indices.shape[0]
        device = history_node_indices.device
        
        # ================================================================
        # STEP 1: COMPUTE UNIFIED EMBEDDING
        # ================================================================
        # For per-node training, we extract the last node embedding
        # Similar to LSTM model approach
        
        # Get node embeddings
        node_emb = self.embedding_pipeline.encode_nodes(
            history_node_indices,
            spatial_coords=None,
            categories=None,
        )  # [batch, seq_len, node_embedding_dim]
        
        # Expand to fusion_dim if needed
        if node_emb.shape[-1] < self.fusion_dim:
            padding = torch.zeros(
                batch_size, node_emb.shape[1], self.fusion_dim - node_emb.shape[-1],
                device=device
            )
            node_emb = torch.cat([node_emb, padding], dim=-1)
        
        # Extract last valid embedding for each sequence
        # Use history_lengths to get correct position
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = (history_lengths - 1).clamp(min=0)
        unified_embedding = node_emb[batch_indices, last_indices]  # [batch, fusion_dim]
        
        # ================================================================
        # STEP 2: PARALLEL BELIEF + DESIRE VAEs
        # ================================================================
        
        # Belief VAE: World model
        belief_out = self.belief_vae(unified_embedding)
        belief_z = belief_out['z']  # [batch, belief_latent_dim]
        
        # Desire VAE: Preferences
        desire_out = self.desire_vae(unified_embedding)
        desire_z = desire_out['z']  # [batch, desire_latent_dim]
        
        # ================================================================
        # STEP 3: HIERARCHICAL INTENTION VAE
        # ================================================================
        
        # Intention VAE: Integrates beliefs + desires + embedding
        intention_out = self.intention_vae(
            belief_z=belief_z,
            desire_z=desire_z,
            embedding=unified_embedding,
        )
        intention_z = intention_out['z']  # [batch, intention_latent_dim]
        
        # ================================================================
        # STEP 4: PREDICTION HEADS
        # ================================================================
        
        # Project intention latent to prediction features
        pred_features = self.feature_projection(intention_z)  # [batch, hidden_dim]
        
        # Compute predictions
        goal_logits = self.goal_head(pred_features)
        nextstep_logits = self.nextstep_head(pred_features)
        category_logits = self.category_head(pred_features)
        
        # ================================================================
        # STEP 5: COMPUTE VAE LOSSES (if requested)
        # ================================================================
        
        outputs = {
            'goal': goal_logits,
            'nextstep': nextstep_logits,
            'category': category_logits,
            'belief_z': belief_z,
            'desire_z': desire_z,
            'intention_z': intention_z,
            'unified_embedding': unified_embedding,
        }
        
        if compute_loss:
            # Belief VAE loss
            belief_loss, belief_recon, belief_kl = vae_loss(
                recon=belief_out['recon'],
                x=unified_embedding,
                mu=belief_out['mu'],
                log_var=belief_out['log_var'],
                beta=self.beta_belief,
                free_bits=free_bits,
                kl_annealing_factor=kl_annealing_factor,
            )
            
            # Desire VAE loss
            desire_loss, desire_recon, desire_kl = vae_loss(
                recon=desire_out['recon'],
                x=unified_embedding,
                mu=desire_out['mu'],
                log_var=desire_out['log_var'],
                beta=self.beta_desire,
                free_bits=free_bits,
                kl_annealing_factor=kl_annealing_factor,
            )
            
            # Intention VAE loss
            intention_loss, intention_recon, intention_kl = vae_loss(
                recon=intention_out['recon'],
                x=intention_out['input'],  # Full concatenated input
                mu=intention_out['mu'],
                log_var=intention_out['log_var'],
                beta=self.beta_intention,
                free_bits=free_bits,
                kl_annealing_factor=kl_annealing_factor,
            )
            
            # Total VAE loss
            total_vae_loss = belief_loss + desire_loss + intention_loss
            
            outputs.update({
                'belief_loss': belief_loss,
                'belief_recon_loss': belief_recon,
                'belief_kl_loss': belief_kl,
                'desire_loss': desire_loss,
                'desire_recon_loss': desire_recon,
                'desire_kl_loss': desire_kl,
                'intention_loss': intention_loss,
                'intention_recon_loss': intention_recon,
                'intention_kl_loss': intention_kl,
                'total_vae_loss': total_vae_loss,
            })
        
        return outputs
    
    def encode_mental_states(
        self,
        history_node_indices: torch.Tensor,
        history_lengths: torch.Tensor,
        agent_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode trajectory to mental state representations (inference mode).
        
        Args:
            history_node_indices: [batch, seq_len]
            history_lengths: [batch]
            agent_ids: [batch] optional
        
        Returns:
            Dict with belief_z, desire_z, intention_z, unified_embedding
        """
        with torch.no_grad():
            outputs = self.forward(
                history_node_indices=history_node_indices,
                history_lengths=history_lengths,
                agent_ids=agent_ids,
                compute_loss=False,
            )
        
        return {
            'belief_z': outputs['belief_z'],
            'desire_z': outputs['desire_z'],
            'intention_z': outputs['intention_z'],
            'unified_embedding': outputs['unified_embedding'],
        }
