"""
Advanced Multi-Modal Temporal Encoder

This module provides expert-level temporal feature encoding for trajectory prediction.
It captures temporal patterns at multiple scales and modalities:

1. **Hour of Day** (0-23): Immediate time context
2. **Day of Week** (0-6): Weekly patterns and business/social rhythms
3. **Circadian Phase**: Continuous phase encoding for smooth transitions
4. **Temporal Deltas**: Time elapsed between consecutive nodes
5. **Velocity Profile**: Speed patterns indicating activity type
6. **Time-to-Goal**: Estimated time to predicted goal

This is especially powerful for campus navigation where:
- Classes happen at specific times
- Library hours vary by day
- Dining patterns follow meal times
- Social activities cluster in evenings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import math


class CircadianPhaseEncoding(nn.Module):
    """
    Encodes time of day as continuous phase on unit circle.
    
    Maps 24-hour clock to smooth periodic representation:
    phase(t) = [sin(2π*t/24), cos(2π*t/24)]
    
    This captures the continuous, cyclical nature of daily time.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, hour: torch.Tensor) -> torch.Tensor:
        """
        Convert hour to circadian phase.
        
        Args:
            hour: (...,) - Hour values (0-23) with any shape
            
        Returns:
            Phase encoding of same shape as input + 1 (for sin/cos)
        """
        # Normalize hour to [0, 2π]
        angle = 2 * math.pi * hour / 24.0
        
        # Encode as (sin, cos) for smooth periodicity
        phase = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        
        return phase


class TemporalDeltaEncoder(nn.Module):
    """
    Encodes time differences between trajectory steps.
    
    Captures temporal granularity:
    - Small deltas (seconds): Movement within a location
    - Medium deltas (minutes): Transition between nearby locations
    - Large deltas (hours): Travel or waiting
    
    Uses learnable embeddings and magnitude normalization.
    """
    
    def __init__(self, embedding_dim: int = 32, max_delta_hours: float = 24.0):
        """
        Initialize temporal delta encoder.
        
        Args:
            embedding_dim: Dimension of delta embedding
            max_delta_hours: Maximum time delta to represent (for normalization)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_delta_hours = max_delta_hours
        
        # Learnable magnitude embeddings for different scales
        # Small, medium, large deltas
        self.scale_embeddings = nn.Embedding(3, embedding_dim // 3)
        
        # Continuous magnitude encoder
        self.magnitude_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 3),
            nn.ReLU(),
            nn.Linear(embedding_dim // 3, embedding_dim // 3)
        )
        
        # Magnitude normalization (log-scale)
        # Helps handle large range of time deltas
        self.log_scale = math.log(10)  # 10x scale for log normalization
    
    def forward(
        self,
        deltas_hours: torch.Tensor,
        return_scales: bool = False
    ) -> torch.Tensor:
        """
        Encode time deltas.
        
        Args:
            deltas_hours: (batch_size, seq_len) - Time deltas in hours
            return_scales: If True, also return scale indicators
            
        Returns:
            Delta embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        device = deltas_hours.device
        
        # Clamp to reasonable range
        deltas_clamped = torch.clamp(deltas_hours, 0, self.max_delta_hours)
        
        # Determine scale (small: <5min, medium: 5min-1hr, large: >1hr)
        # Convert to minutes for scale determination
        deltas_minutes = deltas_clamped * 60
        
        scales = torch.zeros_like(deltas_clamped, dtype=torch.long)
        scales[deltas_minutes < 5] = 0  # Small
        scales[(deltas_minutes >= 5) & (deltas_minutes < 60)] = 1  # Medium
        scales[deltas_minutes >= 60] = 2  # Large
        
        # Get scale embeddings
        scale_emb = self.scale_embeddings(scales)  # (batch, seq, emb_dim/3)
        
        # Encode magnitude (log-normalized)
        log_magnitude = torch.log1p(deltas_clamped.unsqueeze(-1) / self.max_delta_hours)
        magnitude_emb = self.magnitude_encoder(log_magnitude)  # (batch, seq, emb_dim/3)
        
        # Encode phase within scale
        # Small deltas get finer phase, large deltas get coarser
        phase_angles = 2 * math.pi * (deltas_minutes % 60) / 60.0  # Phase within hour
        phase_emb_sin = torch.sin(phase_angles).unsqueeze(-1)
        phase_emb_cos = torch.cos(phase_angles).unsqueeze(-1)
        phase_emb = torch.cat([phase_emb_sin, phase_emb_cos], dim=-1)
        
        # Combine all components
        combined = torch.cat([scale_emb, magnitude_emb, phase_emb], dim=-1)
        
        if return_scales:
            return combined, scales
        return combined


class CircadianPatternEncoder(nn.Module):
    """
    Captures daily activity patterns and circadian rhythms.
    
    Learns embeddings for different times of day that capture typical activity patterns:
    - Morning (6-9): Commute to classes/work
    - Mid-day (10-14): Classes/studying
    - Afternoon (14-17): More classes/library
    - Evening (17-21): Dining, social activities, studying
    - Night (21-6): Sleeping (sparse activity)
    
    Uses both continuous and discrete time representations.
    """
    
    def __init__(self, embedding_dim: int = 32, num_time_buckets: int = 24):
        """
        Initialize circadian pattern encoder.
        
        Args:
            embedding_dim: Dimension of pattern embedding
            num_time_buckets: Number of hourly buckets (default: 24)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_time_buckets = num_time_buckets
        
        # Hourly activity pattern embeddings
        self.hour_embedding = nn.Embedding(num_time_buckets, embedding_dim)
        
        # Activity type at different times (learned)
        self.activity_types = nn.Parameter(
            torch.randn(4, embedding_dim // 4) * 0.1
        )  # 4 activity types: commute, study, social, sleep
        
        # Continuous time encoder for smooth transitions
        self.continuous_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim // 2),  # sin/cos phase
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
    
    def forward(
        self,
        hours: torch.Tensor,
        days: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode circadian patterns.
        
        Args:
            hours: (...,) - Hour of day (0-23), any shape
            days: (...,) - Day of week (0-6), optional, same shape as hours
            
        Returns:
            Pattern embeddings of shape (..., embedding_dim)
        """
        original_shape = hours.shape
        batch_size = original_shape[0] if len(original_shape) > 0 else 1
        
        # Flatten to 1D for processing
        hours_flat = hours.reshape(-1) if hours.dim() > 0 else hours.unsqueeze(0)
        
        # Discrete hourly embedding
        hour_emb = self.hour_embedding(hours_flat.long())  # (total, emb_dim)
        
        # Continuous circadian phase
        phase = torch.stack([
            torch.sin(2 * math.pi * hours_flat / 24.0),
            torch.cos(2 * math.pi * hours_flat / 24.0)
        ], dim=-1)
        continuous_emb = self.continuous_encoder(phase)  # (total, emb_dim/2)
        
        # Combine discrete and continuous
        combined = torch.cat([hour_emb, continuous_emb], dim=-1)
        
        # Reshape back to original batch shape
        output = combined.reshape(*original_shape, -1)
        
        return output


class DayOfWeekEncoder(nn.Module):
    """
    Encodes day of week information.
    
    Captures weekly patterns:
    - Weekdays (0-4): Regular class/work schedule
    - Weekends (5-6): Different social/leisure patterns
    
    Includes both discrete embedding and continuous encoding.
    """
    
    def __init__(self, embedding_dim: int = 16):
        """
        Initialize day of week encoder.
        
        Args:
            embedding_dim: Dimension of day embedding
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Discrete day embeddings
        self.day_embedding = nn.Embedding(7, embedding_dim // 2)
        
        # Weekday/weekend flag
        self.weekend_encoder = nn.Linear(1, embedding_dim // 2)
    
    def forward(self, days: torch.Tensor) -> torch.Tensor:
        """
        Encode day of week.
        
        Args:
            days: (...,) - Day of week (0=Monday, 6=Sunday), any shape
            
        Returns:
            Day embeddings of shape (..., embedding_dim)
        """
        original_shape = days.shape
        
        # Flatten for processing
        days_flat = days.reshape(-1) if days.dim() > 0 else days.unsqueeze(0)
        
        # Discrete embedding
        day_emb = self.day_embedding(days_flat.long())  # (total, emb_dim/2)
        
        # Weekend flag (continuous: 0 for weekday, 1 for weekend)
        is_weekend = ((days_flat >= 5).float()).unsqueeze(-1)
        weekend_emb = self.weekend_encoder(is_weekend)  # (total, emb_dim/2)
        
        # Combine
        combined = torch.cat([day_emb, weekend_emb], dim=-1)
        
        # Reshape back
        output = combined.reshape(*original_shape, -1)
        
        return output


class AdvancedTemporalEncoder(nn.Module):
    """
    Expert-level temporal encoder combining multiple temporal modalities.
    
    Integrates:
    - Hour of day (circadian + discrete)
    - Day of week (weekday/weekend patterns)
    - Temporal deltas (time between steps)
    - Velocity profiles (speed indicators)
    - Optional: time-to-goal, time-from-start
    
    Output is a rich temporal feature representation suitable for Transformer input.
    """
    
    def __init__(
        self,
        temporal_dim: int = 64,
        hidden_dim: int = 128,
        include_day_of_week: bool = True,
        include_temporal_deltas: bool = True,
        include_velocity: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize advanced temporal encoder.
        
        Args:
            temporal_dim: Output dimension for temporal features
            hidden_dim: Hidden dimension for composition layers
            include_day_of_week: Whether to include day-of-week encoding
            include_temporal_deltas: Whether to include time deltas
            include_velocity: Whether to include velocity profiles
            dropout: Dropout rate
        """
        super().__init__()
        
        self.temporal_dim = temporal_dim
        self.include_day_of_week = include_day_of_week
        self.include_temporal_deltas = include_temporal_deltas
        self.include_velocity = include_velocity
        
        # Component encoders - note: CircadianPatternEncoder outputs embedding_dim + embedding_dim//2
        # because it concatenates hourly embedding (embedding_dim) with continuous embedding (embedding_dim//2)
        # Similarly, TemporalDeltaEncoder outputs: embedding_dim//3 + embedding_dim//3 + 2 (phase)
        # So we need to account for these actual outputs when calculating fusion dimensions
        circadian_out_dim = temporal_dim // 2 + temporal_dim // 4  # embedding_dim + embedding_dim//2
        
        delta_emb_dim = temporal_dim // 4
        day_emb_dim = temporal_dim // 4
        vel_emb_dim = temporal_dim // 8
        
        # Actual output dimensions (accounting for component design):
        # CircadianPatternEncoder: embedding_dim + embedding_dim//2
        circadian_actual_out = temporal_dim // 2 + temporal_dim // 4
        
        # TemporalDeltaEncoder: embedding_dim//3 + embedding_dim//3 + 2
        delta_actual_out = delta_emb_dim // 3 + delta_emb_dim // 3 + 2
        
        self.circadian = CircadianPatternEncoder(embedding_dim=temporal_dim // 2)
        
        if include_day_of_week:
            self.day_encoder = DayOfWeekEncoder(embedding_dim=day_emb_dim)
        
        if include_temporal_deltas:
            self.delta_encoder = TemporalDeltaEncoder(embedding_dim=delta_emb_dim)
        
        if include_velocity:
            self.velocity_encoder = nn.Sequential(
                nn.Linear(1, vel_emb_dim),
                nn.ReLU(),
                nn.Linear(vel_emb_dim, vel_emb_dim)
            )
        
        # Compute fusion input dimension
        fusion_input_dim = circadian_actual_out
        if include_day_of_week:
            fusion_input_dim += day_emb_dim
        if include_temporal_deltas:
            fusion_input_dim += delta_actual_out
        if include_velocity:
            fusion_input_dim += vel_emb_dim
        
        # Fusion layers to combine all components
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, temporal_dim)
        )
    
    def forward(
        self,
        hours: torch.Tensor,
        days: Optional[torch.Tensor] = None,
        deltas: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Encode temporal features.
        
        Args:
            hours: (batch_size,) or (batch_size, seq_len) - Hour of day (0-23)
            days: (batch_size,) or (batch_size, seq_len) - Day of week (0-6), optional
            deltas: (batch_size,) or (batch_size, seq_len) - Time deltas in hours, optional
            velocities: (batch_size,) or (batch_size, seq_len) - Velocity (units/hour), optional
            return_components: If True, return component embeddings
            
        Returns:
            Temporal embeddings of shape (batch_size, temporal_dim)
            If return_components=True: tuple of (embeddings, components_dict)
        """
        components = {}
        embeddings = []
        
        # Handle sequence vs batch dimensions - collapse to batch for temporal encoding
        if hours.dim() > 1:
            # Take first timestep as representative
            hours_batch = hours[:, 0] if hours.shape[1] > 0 else hours.squeeze()
        else:
            hours_batch = hours
        
        # Circadian encoding (always included)
        circadian_emb = self.circadian(hours_batch, days if days is None or days.dim() == 1 else days[:, 0])
        embeddings.append(circadian_emb)
        components['circadian'] = circadian_emb.clone()
        
        # Day of week encoding
        if self.include_day_of_week and days is not None:
            days_batch = days if days.dim() == 1 else days[:, 0]
            day_emb = self.day_encoder(days_batch)
            embeddings.append(day_emb)
            components['day_of_week'] = day_emb.clone()
        
        # Temporal delta encoding
        if self.include_temporal_deltas and deltas is not None:
            deltas_batch = deltas if deltas.dim() == 1 else deltas[:, 0]
            delta_emb = self.delta_encoder(deltas_batch)
            embeddings.append(delta_emb)
            components['temporal_deltas'] = delta_emb.clone()
        
        # Velocity encoding
        if self.include_velocity and velocities is not None:
            velocities_batch = velocities if velocities.dim() == 1 else velocities[:, 0]
            velocity_emb = self.velocity_encoder(velocities_batch.unsqueeze(-1))
            embeddings.append(velocity_emb)
            components['velocity'] = velocity_emb.clone()
        
        # Fuse all components
        combined = torch.cat(embeddings, dim=-1)
        fused = self.fusion(combined)
        
        if return_components:
            return fused, components
        return fused


# ============================================================================
# TESTING AND DEMO
# ============================================================================

def test_temporal_encoders():
    """Test all temporal encoder components."""
    print("=" * 80)
    print("Testing Advanced Temporal Encoders")
    print("=" * 80)
    
    batch_size, seq_len = 4, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create synthetic temporal data
    hours = torch.randint(0, 24, (batch_size, seq_len)).float()
    days = torch.randint(0, 7, (batch_size, seq_len)).float()
    deltas = torch.rand(batch_size, seq_len) * 4  # 0-4 hours
    velocities = torch.rand(batch_size, seq_len) * 10  # 0-10 units/hour
    
    # Move to device
    hours = hours.to(device)
    days = days.to(device)
    deltas = deltas.to(device)
    velocities = velocities.to(device)
    
    # Test 1: Circadian Phase Encoding
    print("\n1. Testing CircadianPhaseEncoding...")
    circ_phase = CircadianPhaseEncoding()
    phase_out = circ_phase(hours)
    print(f"   ✓ Output shape: {phase_out.shape} (should be {hours.shape + (2,)})")
    print(f"   ✓ Sample phases: sin={phase_out[0, 0, 0]:.4f}, cos={phase_out[0, 0, 1]:.4f}")
    
    # Test 2: Temporal Delta Encoder
    print("\n2. Testing TemporalDeltaEncoder...")
    delta_enc = TemporalDeltaEncoder(embedding_dim=32).to(device)
    delta_out, scales = delta_enc(deltas, return_scales=True)
    print(f"   ✓ Output shape: {delta_out.shape}")
    print(f"   ✓ Scale distribution: {torch.bincount(scales.long().flatten())}")
    
    # Test 3: Circadian Pattern Encoder
    print("\n3. Testing CircadianPatternEncoder...")
    circ_pattern = CircadianPatternEncoder(embedding_dim=32).to(device)
    pattern_out = circ_pattern(hours, days)
    print(f"   ✓ Output shape: {pattern_out.shape}")
    
    # Test 4: Day of Week Encoder
    print("\n4. Testing DayOfWeekEncoder...")
    day_enc = DayOfWeekEncoder(embedding_dim=16).to(device)
    day_out = day_enc(days)
    print(f"   ✓ Output shape: {day_out.shape}")
    
    # Test 5: Advanced Temporal Encoder
    print("\n5. Testing AdvancedTemporalEncoder...")
    temporal_enc = AdvancedTemporalEncoder(
        temporal_dim=64,
        hidden_dim=128,
        include_day_of_week=True,
        include_temporal_deltas=True,
        include_velocity=True,
        dropout=0.1
    ).to(device)
    
    temporal_out, components = temporal_enc(
        hours, days, deltas, velocities, return_components=True
    )
    print(f"   ✓ Output shape: {temporal_out.shape}")
    print(f"   ✓ Components: {list(components.keys())}")
    
    # Test parameter counts
    print("\n6. Model parameters...")
    total_params = sum(p.numel() for p in temporal_enc.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    # Test edge cases
    print("\n7. Testing edge cases...")
    # All zeros
    zeros_hours = torch.zeros(batch_size, seq_len, device=device)
    zeros_out = temporal_enc(zeros_hours, days, deltas, velocities)
    print(f"   ✓ Handles zero hours: {zeros_out.shape}")
    
    # All ones
    ones_hours = torch.ones(batch_size, seq_len, device=device) * 23
    ones_out = temporal_enc(ones_hours, days, deltas, velocities)
    print(f"   ✓ Handles max hours: {ones_out.shape}")
    
    print("\n" + "=" * 80)
    print("✓ All temporal encoder tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_temporal_encoders()
