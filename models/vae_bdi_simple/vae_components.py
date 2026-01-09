"""
VAE COMPONENTS FOR BDI HIERARCHICAL VARIATIONAL AUTOENCODER

This module contains the core variational autoencoder building blocks
for the BDI (Belief-Desire-Intention) mental state representation model.

COMPONENTS:
1. VAEEncoder: Encodes input to latent distribution (mean + log_var)
2. VAEDecoder: Reconstructs input from latent samples
3. Individual VAE modules for Beliefs, Desires, and Intentions

THEORY OF MIND MAPPING:
- Belief VAE: Learns latent representations of world model (spatial beliefs)
- Desire VAE: Learns latent representations of preferences (goals/values)
- Intention VAE: Learns latent representations of goal-directed plans
  (conditioned on beliefs + desires)

VAE LOSS:
L = Reconstruction Loss + β * KL Divergence
- Reconstruction: How well can we reconstruct from latent?
- KL Divergence: How close is latent distribution to prior N(0,1)?
- β: Weight for KL term (can use β-VAE for disentanglement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class VAEEncoder(nn.Module):
    """
    VAE Encoder: Maps input to latent distribution parameters (μ, log σ²).
    
    Architecture:
    Input → [Hidden Layers] → [μ head, log σ² head]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Build hidden layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: [batch, input_dim] input features
        
        Returns:
            mu: [batch, latent_dim] mean of latent distribution
            log_var: [batch, latent_dim] log variance of latent distribution
        """
        # Process through hidden layers
        h = self.hidden_layers(x)
        
        # Compute distribution parameters
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        return mu, log_var


class VAEDecoder(nn.Module):
    """
    VAE Decoder: Reconstructs input from latent samples.
    
    Architecture:
    Latent z → [Hidden Layers] → Reconstruction
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of reconstructed output
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build hidden layers
        layers = []
        current_dim = latent_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Reconstruction head
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent sample to reconstruction.
        
        Args:
            z: [batch, latent_dim] latent samples
        
        Returns:
            recon: [batch, output_dim] reconstructed output
        """
        h = self.hidden_layers(z)
        recon = self.fc_out(h)
        return recon


class BeliefVAE(nn.Module):
    """
    Belief VAE: Learns latent representations of world model.
    
    Beliefs represent:
    - Spatial knowledge (where things are)
    - Graph structure understanding
    - Environmental context
    
    Input: Unified embedding (spatial + temporal context)
    Output: Belief latent representation
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
        self.latent_dim = latent_dim
        
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
        
        Args:
            mu: [batch, latent_dim] mean
            log_var: [batch, latent_dim] log variance
        
        Returns:
            z: [batch, latent_dim] sampled latent
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Belief VAE.
        
        Args:
            x: [batch, input_dim] unified embedding
        
        Returns:
            Dict with:
                - 'mu': [batch, latent_dim] latent mean
                - 'log_var': [batch, latent_dim] latent log variance
                - 'z': [batch, latent_dim] sampled latent
                - 'recon': [batch, input_dim] reconstruction
        """
        # Encode to latent distribution
        mu, log_var = self.encoder(x)
        
        # Sample latent
        z = self.reparameterize(mu, log_var)
        
        # Reconstruct
        recon = self.decoder(z)
        
        return {
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'recon': recon,
        }


class DesireVAE(nn.Module):
    """
    Desire VAE: Learns latent representations of preferences.
    
    Desires represent:
    - Agent preferences (category preferences)
    - Goal values (what they want)
    - Motivational states
    
    Input: Unified embedding (agent + behavioral context)
    Output: Desire latent representation
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
        self.latent_dim = latent_dim
        
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Desire VAE.
        
        Args:
            x: [batch, input_dim] unified embedding
        
        Returns:
            Dict with mu, log_var, z, recon
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        
        return {
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'recon': recon,
        }


class IntentionVAE(nn.Module):
    """
    Intention VAE: Learns latent representations of goal-directed plans.
    
    Intentions represent:
    - Goal-directed action plans
    - Integration of beliefs + desires
    - Behavioral policies
    
    Input: Concatenation of [belief_z, desire_z, unified_embedding]
    Output: Intention latent representation
    
    This is the top-level VAE in the hierarchy that conditions on
    both belief and desire representations.
    """
    
    def __init__(
        self,
        belief_latent_dim: int,
        desire_latent_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            belief_latent_dim: Dimension of belief latent space
            desire_latent_dim: Dimension of desire latent space
            embedding_dim: Dimension of unified embedding (skip connection)
            hidden_dim: Hidden dimension
            latent_dim: Dimension of intention latent space
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input is concatenation of belief, desire, and embedding
        input_dim = belief_latent_dim + desire_latent_dim + embedding_dim
        
        self.belief_latent_dim = belief_latent_dim
        self.desire_latent_dim = desire_latent_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,  # Reconstruct full input
            num_layers=num_layers,
            dropout=dropout,
        )
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        belief_z: torch.Tensor,
        desire_z: torch.Tensor,
        embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Intention VAE.
        
        Args:
            belief_z: [batch, belief_latent_dim] belief latent
            desire_z: [batch, desire_latent_dim] desire latent
            embedding: [batch, embedding_dim] unified embedding (skip connection)
        
        Returns:
            Dict with mu, log_var, z, recon
        """
        # Concatenate all inputs
        x = torch.cat([belief_z, desire_z, embedding], dim=-1)
        
        # Encode to intention latent
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        
        # Reconstruct
        recon = self.decoder(z)
        
        return {
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'recon': recon,
            'input': x,  # Save input for reconstruction loss
        }


def free_bits_kl(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    free_bits: float = 0.5,
) -> torch.Tensor:
    """
    Compute KL divergence with free bits to prevent KL collapse.
    
    Free bits enforces a minimum KL per latent dimension, preventing the
    posterior from collapsing to the prior. This ensures each latent dimension
    encodes at least free_bits nats of information.
    
    Args:
        mu: [batch, latent_dim] latent mean
        log_var: [batch, latent_dim] latent log variance
        free_bits: Minimum KL per dimension (default: 0.5 nats)
    
    Returns:
        kl_loss: Scalar KL divergence loss
    """
    # KL per dimension: KL(q(z|x) || p(z)) where p(z) = N(0,1)
    # KL_i = -0.5 * (1 + log_var_i - mu_i^2 - exp(log_var_i))
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    
    # Clamp each dimension to minimum free_bits
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    
    # Sum over latent dimensions, mean over batch
    kl_loss = kl_per_dim.sum(dim=-1).mean()
    
    return kl_loss


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.5,
    kl_annealing_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss: Reconstruction + β * KL Divergence (with free bits and annealing).
    
    Args:
        recon: [batch, dim] reconstructed output
        x: [batch, dim] original input
        mu: [batch, latent_dim] latent mean
        log_var: [batch, latent_dim] latent log variance
        beta: Weight for KL divergence term (β-VAE)
        free_bits: Minimum KL per dimension to prevent collapse (default: 0.5)
        kl_annealing_factor: Multiplicative factor for KL annealing (0→1 over warmup)
    
    Returns:
        total_loss: Scalar total loss
        recon_loss: Scalar reconstruction loss
        kl_loss: Scalar KL divergence loss (before annealing)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL divergence with free bits to prevent collapse
    kl_loss = free_bits_kl(mu, log_var, free_bits=free_bits)
    
    # Total loss with β-VAE weighting and annealing
    total_loss = recon_loss + (beta * kl_annealing_factor) * kl_loss
    
    return total_loss, recon_loss, kl_loss
