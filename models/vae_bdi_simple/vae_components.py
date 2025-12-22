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


def check_encoder_health(mu: torch.Tensor, log_var: torch.Tensor, name: str = "VAE") -> Dict[str, float]:
    """
    Diagnostic function to check if VAE encoder is healthy or collapsing.
    
    Signs of collapse:
    - mu variance near zero (all latents same)
    - log_var near large negative (variance -> 0, deterministic)
    - std (exp(0.5 * log_var)) near zero
    
    Healthy VAE:
    - mu variance > 0.01 (latents differ across samples)
    - log_var in range [-2, 2] (std in [0.37, 2.72])
    - KL per dim > 0.1 (encodes information)
    
    Args:
        mu: [batch, latent_dim] latent means
        log_var: [batch, latent_dim] latent log variances
        name: Name for logging
    
    Returns:
        Dict with diagnostic metrics
    """
    with torch.no_grad():
        # Compute statistics
        mu_mean = mu.mean().item()
        mu_std = mu.std().item()
        mu_var = mu.var().item()
        
        log_var_mean = log_var.mean().item()
        log_var_std = log_var.std().item()
        
        # Compute actual std from log_var
        std = torch.exp(0.5 * log_var)
        std_mean = std.mean().item()
        std_std = std.std().item()
        
        # KL per dimension
        kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_mean = kl_per_dim.mean().item()
        
        # Check for collapse
        is_collapsed = (
            mu_var < 0.01 or  # All latents same
            std_mean < 0.1 or  # Variance collapsed
            kl_mean < 0.01     # No information encoded
        )
        
        diagnostics = {
            f'{name}_mu_mean': mu_mean,
            f'{name}_mu_std': mu_std,
            f'{name}_mu_var': mu_var,
            f'{name}_log_var_mean': log_var_mean,
            f'{name}_log_var_std': log_var_std,
            f'{name}_std_mean': std_mean,
            f'{name}_std_std': std_std,
            f'{name}_kl_per_dim_mean': kl_mean,
            f'{name}_is_collapsed': 1.0 if is_collapsed else 0.0,
        }
        
        return diagnostics


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
        
        # Initialize weights to prevent collapse
        # Small initialization for mu (encourages exploration)
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.zeros_(self.fc_mu.bias)
        
        # Initialize log_var to encode uncertainty (not collapse to deterministic)
        # Start with log_var ≈ 0 (variance ≈ 1) for healthy exploration
        nn.init.xavier_normal_(self.fc_log_var.weight, gain=0.01)
        nn.init.zeros_(self.fc_log_var.bias)
    
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
        
        # Clamp log_var to prevent numerical instability and collapse
        # Prevent variance from becoming too small (collapse) or too large (instability)
        # log_var in [-10, 2] means std in [exp(-5), exp(1)] = [0.0067, 2.718]
        log_var = torch.clamp(log_var, min=-10.0, max=2.0)
        
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


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,
    recon_scale: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss: Reconstruction + β * KL Divergence with free bits.
    
    Free bits prevents posterior collapse by setting a minimum KL per dimension.
    The model is not penalized for KL until it exceeds the threshold.
    
    CRITICAL: Free bits is applied PER DIMENSION, not total KL!
    This forces the model to encode at least free_bits nats of information
    in EACH latent dimension, preventing collapse to uninformative posterior.
    
    Args:
        recon: [batch, dim] reconstructed output
        x: [batch, dim] original input
        mu: [batch, latent_dim] latent mean
        log_var: [batch, latent_dim] latent log variance
        beta: Weight for KL divergence term (β-VAE)
               Set to 0.0 to disable KL penalty (free bits provides floor)
        free_bits: Free bits threshold in nats per dimension.
                   Recommended: 0.5-2.0 nats/dim to force information encoding.
                   Each dimension must encode at least this much info before penalty.
        recon_scale: Scale factor for reconstruction loss to prevent it from being too small.
                     Default: 100.0 (makes recon loss comparable to KL)
    
    Returns:
        total_loss: Scalar total loss
        recon_loss: Scalar reconstruction loss (unscaled, for monitoring)
        kl_loss: Scalar KL divergence loss (after free bits, used in optimization)
        kl_raw: Scalar raw KL divergence (before free bits, for monitoring)
    """
    # Reconstruction loss (MSE)
    recon_loss_raw = F.mse_loss(recon, x, reduction='mean')
    
    # Scale reconstruction loss to make it significant
    # This prevents the model from ignoring reconstruction
    recon_loss = recon_loss_raw * recon_scale
    
    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,1)
    # KL per dimension: KL_d = -0.5 * (1 + log_var_d - mu_d^2 - exp(log_var_d))
    # This gives KL for each latent dimension separately
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # [batch, latent_dim]
    
    # Total KL per sample (sum over dimensions)
    kl_per_sample = kl_per_dim.sum(dim=-1)  # [batch]
    
    # Raw KL (before free bits) for monitoring
    kl_raw = kl_per_sample.mean()
    
    # Apply free bits PER DIMENSION to prevent collapse
    # This is CRITICAL: we threshold each dimension independently
    if free_bits > 0.0:
        # Clamp each dimension's KL to be at least 0 after subtracting threshold
        # max(0, KL_d - free_bits) for each dimension
        kl_per_dim_clamped = torch.clamp(kl_per_dim - free_bits, min=0.0)  # [batch, latent_dim]
        
        # Sum over dimensions to get total clamped KL per sample
        kl_per_sample = kl_per_dim_clamped.sum(dim=-1)  # [batch]
    
    kl_loss = kl_per_sample.mean()
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    # Return unscaled recon_loss for monitoring
    return total_loss, recon_loss_raw, kl_loss, kl_raw
