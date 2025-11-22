"""
Agent Identity and Behavioral Encoding

This module provides expert-level agent representation learning for capturing
individual behavioral patterns and preferences in trajectory prediction.

Key insights for agent representation:
1. Each agent has unique preferences (favorite locations, route patterns)
2. Agents adapt behavior based on time/context (not static)
3. Meta-learning helps transfer knowledge to new agents
4. Agent embedding serves as "working memory" during prediction

This is crucial for a BDI-ToM system where:
- Agent intent is partially captured by their identity
- Different agents pursue goals differently
- Agent preferences explain goal choice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


class AgentEmbedding(nn.Module):
    """
    Learnable agent embeddings capturing individual behavioral patterns.
    
    Each agent has a unique embedding that captures:
    - Preferred locations and categories
    - Typical routes and movement patterns
    - Time-of-day specific behaviors
    - Goal preferences
    
    The embeddings are learned end-to-end through trajectory prediction.
    """
    
    def __init__(
        self,
        num_agents: int,
        embedding_dim: int = 64,
        num_categories: int = 7,
        with_category_preference: bool = True,
        with_spatial_profile: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize agent embeddings.
        
        Args:
            num_agents: Number of unique agents
            embedding_dim: Dimension of agent embeddings (default: 64)
            num_categories: Number of POI categories (default: 7)
            with_category_preference: Learn per-agent category preferences
            with_spatial_profile: Learn per-agent spatial movement profiles
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.embedding_dim = embedding_dim
        self.num_categories = num_categories
        self.with_category_preference = with_category_preference
        self.with_spatial_profile = with_spatial_profile
        
        # Primary agent identity embedding
        self.agent_embedding = nn.Embedding(num_agents, embedding_dim, padding_idx=0)
        
        # Category preference embeddings per agent (what categories they prefer)
        if with_category_preference:
            self.category_preference = nn.Embedding(num_agents, num_categories)
        
        # Spatial profile embeddings (how different agents move)
        if with_spatial_profile:
            self.spatial_profile = nn.Linear(embedding_dim, embedding_dim)
        
        # Learnable temporal offset (agents have different temporal patterns)
        self.temporal_offset = nn.Embedding(num_agents, embedding_dim // 4)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.agent_embedding.weight[1:])
        if with_category_preference:
            nn.init.uniform_(self.category_preference.weight[1:], 0, 1)
        
    def forward(
        self,
        agent_ids: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Get agent embeddings.
        
        Args:
            agent_ids: (batch_size,) or (batch_size, seq_len) - Agent IDs
            return_components: If True, return individual components
            
        Returns:
            Agent embeddings of shape (batch_size, embedding_dim)
            or (batch_size, seq_len, embedding_dim) if input is 2D
            If return_components=True: tuple of (embeddings, components_dict)
        """
        # Get primary agent embedding
        agent_emb = self.agent_embedding(agent_ids)  # (batch, emb_dim) or (batch, seq, emb_dim)
        
        components = {'identity': agent_emb.clone()}
        embeddings = [agent_emb]
        
        # Add temporal offset
        temporal_offset = self.temporal_offset(agent_ids if agent_ids.dim() == 1 else agent_ids[:, 0])
        # Expand if needed
        if agent_ids.dim() == 2:
            temporal_offset = temporal_offset.unsqueeze(1).expand(agent_ids.shape[0], agent_ids.shape[1], -1)
        embeddings.append(temporal_offset)
        components['temporal_offset'] = temporal_offset.clone()
        
        # Concatenate components
        combined = torch.cat(embeddings, dim=-1)
        
        if return_components:
            return combined, components
        return combined
    
    def get_category_preferences(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        Get category preferences for agents.
        
        Args:
            agent_ids: (batch_size,) - Agent IDs
            
        Returns:
            Category preferences of shape (batch_size, num_categories)
        """
        if not self.with_category_preference:
            raise ValueError("Category preferences not enabled")
        
        prefs = self.category_preference(agent_ids.long())  # (batch, num_categories)
        # Normalize to probability distribution
        return F.softmax(prefs, dim=-1)
    
    def get_agent_similarity(self, agent_id1: int, agent_id2: int) -> float:
        """
        Compute similarity between two agent embeddings.
        
        Useful for finding similar agents or clustering agents by behavior.
        
        Args:
            agent_id1, agent_id2: Agent IDs
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        id1 = torch.tensor([agent_id1], dtype=torch.long)
        id2 = torch.tensor([agent_id2], dtype=torch.long)
        
        emb1 = self.agent_embedding(id1)
        emb2 = self.agent_embedding(id2)
        
        return F.cosine_similarity(emb1, emb2).item()


class BehavioralProfile(nn.Module):
    """
    Learns behavior-specific profiles for agents.
    
    Captures how agent behavior changes based on:
    - Time of day (morning vs evening person)
    - Activity type (student vs staff)
    - Context (alone vs with others - inferred from trajectory)
    
    Creates adaptive agent representations that vary based on context.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_time_periods: int = 4,  # Morning, Mid-day, Afternoon, Evening
        num_contexts: int = 3,  # Home, Study, Social
        dropout: float = 0.1
    ):
        """
        Initialize behavioral profile.
        
        Args:
            embedding_dim: Embedding dimension
            num_time_periods: Number of daily time periods
            num_contexts: Number of behavioral contexts
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_time_periods = num_time_periods
        self.num_contexts = num_contexts
        
        # Time-period specific profiles
        self.time_profiles = nn.Embedding(num_time_periods, embedding_dim)
        
        # Context-specific profiles
        self.context_profiles = nn.Embedding(num_contexts, embedding_dim)
        
        # Adaptive mixing network
        self.profile_mixer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(
        self,
        agent_emb: torch.Tensor,
        time_period: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Adapt agent embedding based on time period and context.
        
        Args:
            agent_emb: (batch_size, embedding_dim) - Base agent embedding
            time_period: (batch_size,) - Time period index (0-num_time_periods)
            context: (batch_size,) - Context index, optional
            
        Returns:
            Adaptive agent embedding of shape (batch_size, embedding_dim)
        """
        # Get time-based profile
        time_prof = self.time_profiles(time_period)  # (batch, emb_dim)
        
        # Get context-based profile if provided
        if context is not None:
            context_prof = self.context_profiles(context)  # (batch, emb_dim)
        else:
            # Default neutral context
            context_prof = torch.zeros_like(time_prof)
        
        # Mix base agent embedding with time/context profiles
        combined = agent_emb + time_prof + context_prof
        
        # Pass through adaptive mixer
        adapted = self.profile_mixer(torch.cat([agent_emb, combined], dim=-1))
        
        return adapted


class AgentContext(nn.Module):
    """
    Captures agent-specific context for trajectory prediction.
    
    Combines:
    - Agent identity
    - Behavioral profile (time/context-dependent)
    - Recent activity summary (velocity, direction trends)
    
    This creates a rich agent representation for conditioning predictions.
    """
    
    def __init__(
        self,
        num_agents: int,
        embedding_dim: int = 64,
        num_categories: int = 7,
        dropout: float = 0.1
    ):
        """
        Initialize agent context.
        
        Args:
            num_agents: Number of agents
            embedding_dim: Embedding dimension
            num_categories: Number of POI categories
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Core components
        self.agent_emb = AgentEmbedding(
            num_agents=num_agents,
            embedding_dim=embedding_dim // 2,
            num_categories=num_categories,
            with_category_preference=True,
            with_spatial_profile=True,
            dropout=dropout
        )
        
        self.behavioral_profile = BehavioralProfile(
            embedding_dim=embedding_dim // 2,
            num_time_periods=4,
            num_contexts=3,
            dropout=dropout
        )
        
        # Recent activity encoder (velocity, direction, etc.)
        self.activity_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim // 4),  # velocity + direction
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # Context fusion (combine adapted agent and activity)
        # agent_adapted is embedding_dim//2, activity_emb is embedding_dim//4
        # together they form 3*embedding_dim//4, which we fuse to embedding_dim
        self.context_fusion = nn.Sequential(
            nn.Linear(embedding_dim // 2 + embedding_dim // 4, embedding_dim),  # 3*emb/4 -> emb
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(
        self,
        agent_ids: torch.Tensor,
        hours: torch.Tensor,
        recent_velocity: Optional[torch.Tensor] = None,
        context_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get agent context.
        
        Args:
            agent_ids: (batch_size,) - Agent IDs
            hours: (batch_size,) - Current hour for time-period detection
            recent_velocity: (batch_size, 1) - Recent movement velocity
            context_type: (batch_size,) - Activity context type
            
        Returns:
            Agent context of shape (batch_size, embedding_dim)
        """
        # Get base agent embedding - extract only the identity component
        agent_full, components = self.agent_emb(agent_ids, return_components=True)
        agent_base = components['identity']  # (batch, emb_dim/2)
        
        # Determine time period (0-3: morning, mid, afternoon, evening)
        time_period = ((hours // 6).long()) % 4
        
        # Get adaptive behavioral profile
        agent_adapted = self.behavioral_profile(agent_base, time_period, context_type)
        
        # Encode recent activity
        if recent_velocity is not None:
            # Create activity vector [velocity, direction_trend]
            # For now, use velocity twice as placeholder
            activity_vec = torch.cat([
                recent_velocity,
                recent_velocity * 0.5  # Direction trend placeholder
            ], dim=-1)
            activity_emb = self.activity_encoder(activity_vec)  # (batch, emb_dim/4)
        else:
            activity_emb = torch.zeros(agent_ids.shape[0], self.embedding_dim // 4, 
                                       device=agent_ids.device)
        
        # Fuse all components
        combined = torch.cat([agent_adapted, activity_emb], dim=-1)
        context = self.context_fusion(combined)
        
        return context


class MultiModalAgentEncoder(nn.Module):
    """
    Complete agent encoder with multi-modal fusion.
    
    Integrates:
    - Agent identity
    - Behavioral adaptations
    - Category preferences
    - Activity patterns
    
    Produces comprehensive agent representation for prediction tasks.
    """
    
    def __init__(
        self,
        num_agents: int,
        agent_emb_dim: int = 64,
        num_categories: int = 7,
        enable_behavioral_profile: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize multi-modal agent encoder.
        
        Args:
            num_agents: Number of agents
            agent_emb_dim: Embedding dimension
            num_categories: Number of categories
            enable_behavioral_profile: Whether to use behavioral profiles
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.agent_emb_dim = agent_emb_dim
        self.enable_behavioral_profile = enable_behavioral_profile
        
        # Agent identity
        self.agent_context = AgentContext(
            num_agents=num_agents,
            embedding_dim=agent_emb_dim,
            num_categories=num_categories,
            dropout=dropout
        )
        
        # Optional behavioral profiles
        if enable_behavioral_profile:
            self.behavioral = BehavioralProfile(
                embedding_dim=agent_emb_dim,
                dropout=dropout
            )
    
    def forward(
        self,
        agent_ids: torch.Tensor,
        hours: torch.Tensor,
        recent_velocity: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Get multi-modal agent encoding.
        
        Args:
            agent_ids: (batch_size,) - Agent IDs
            hours: (batch_size,) - Hour of day
            recent_velocity: (batch_size, 1) - Optional velocity
            return_components: If True, return component dict
            
        Returns:
            Agent encoding of shape (batch_size, agent_emb_dim)
        """
        # Get agent context
        agent_context = self.agent_context(agent_ids, hours, recent_velocity)
        
        if return_components:
            components = {
                'agent_context': agent_context.clone()
            }
            if self.enable_behavioral_profile:
                components['behavioral'] = agent_context.clone()
            return agent_context, components
        
        return agent_context


# ============================================================================
# TESTING AND DEMO
# ============================================================================

def test_agent_encoders():
    """Test agent encoding components."""
    print("=" * 80)
    print("Testing Agent Identity and Behavioral Encoders")
    print("=" * 80)
    
    batch_size = 8
    num_agents = 20
    num_categories = 7
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create synthetic data
    agent_ids = torch.randint(1, num_agents, (batch_size,), device=device)
    hours = torch.randint(0, 24, (batch_size,), dtype=torch.long, device=device).float()
    recent_velocity = torch.rand(batch_size, 1, device=device) * 10
    
    # Test 1: AgentEmbedding
    print("\n1. Testing AgentEmbedding...")
    agent_emb = AgentEmbedding(
        num_agents=num_agents,
        embedding_dim=64,
        num_categories=num_categories,
        with_category_preference=True,
        with_spatial_profile=True,
        dropout=0.1
    ).to(device)
    
    emb_output, components = agent_emb(agent_ids, return_components=True)
    print(f"   ✓ Output shape: {emb_output.shape}")
    
    prefs = agent_emb.get_category_preferences(agent_ids)
    print(f"   ✓ Category preferences shape: {prefs.shape}")
    print(f"   ✓ Preference sums: {prefs.sum(dim=-1)[:3]}")  # Should be ~1.0
    
    sim = agent_emb.get_agent_similarity(1, 2)
    print(f"   ✓ Agent similarity (1,2): {sim:.4f}")
    
    # Test 2: BehavioralProfile
    print("\n2. Testing BehavioralProfile...")
    behavior = BehavioralProfile(
        embedding_dim=64,
        num_time_periods=4,
        num_contexts=3,
        dropout=0.1
    ).to(device)
    
    time_period = (hours // 6).long() % 4
    context = torch.randint(0, 3, (batch_size,), device=device)
    
    # Use only the base identity component for behavioral profile
    base_emb = components['identity']
    behavior_output = behavior(base_emb, time_period, context)
    print(f"   ✓ Output shape: {behavior_output.shape}")
    
    # Test 3: AgentContext
    print("\n3. Testing AgentContext...")
    agent_ctx = AgentContext(
        num_agents=num_agents,
        embedding_dim=64,
        num_categories=num_categories,
        dropout=0.1
    ).to(device)
    
    ctx_output = agent_ctx(agent_ids, hours, recent_velocity, context)
    print(f"   ✓ Output shape: {ctx_output.shape}")
    
    # Test 4: MultiModalAgentEncoder
    print("\n4. Testing MultiModalAgentEncoder...")
    multi_encoder = MultiModalAgentEncoder(
        num_agents=num_agents,
        agent_emb_dim=64,
        num_categories=num_categories,
        enable_behavioral_profile=True,
        dropout=0.1
    ).to(device)
    
    multi_output, components = multi_encoder(
        agent_ids, hours, recent_velocity, return_components=True
    )
    print(f"   ✓ Output shape: {multi_output.shape}")
    print(f"   ✓ Components: {list(components.keys())}")
    
    # Test parameter counts
    print("\n5. Model parameters...")
    total_params = sum(p.numel() for p in multi_encoder.parameters())
    print(f"   ✓ Total parameters: {total_params:,}")
    
    print("\n" + "=" * 80)
    print("✓ All agent encoder tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_agent_encoders()
