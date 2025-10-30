"""
GLM Memory Bank for geometric information.
Stores spatial features and geometric priors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class GLMMemoryBank(nn.Module):
    """
    GLM Memory Bank.
    
    Stores:
    1. Spatial features from previous frames
    2. Object locations and scales
    3. Motion patterns
    
    Enables temporal consistency and motion prediction.
    
    Args:
        feature_dim: Feature dimension
        memory_size: Maximum number of frames to store
        use_static: Whether to use static query memory
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        memory_size: int = 50,
        use_static: bool = False,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.use_static = use_static
        
        # Memory buffers
        self.register_buffer(
            'feature_memory',
            torch.zeros(memory_size, feature_dim)
        )
        self.register_buffer(
            'location_memory',
            torch.zeros(memory_size, 2)  # (x, y) center
        )
        self.register_buffer(
            'scale_memory',
            torch.zeros(memory_size, 2)  # (width, height)
        )
        self.register_buffer(
            'timestamp_memory',
            torch.zeros(memory_size, dtype=torch.long)
        )
        self.register_buffer(
            'confidence_memory',
            torch.zeros(memory_size)
        )
        
        # Pointers
        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        
        # Static query memory (for query frame)
        if use_static:
            self.register_buffer(
                'static_query_feature',
                torch.zeros(feature_dim)
            )
            self.register_buffer(
                'static_query_location',
                torch.zeros(2)
            )
            self.register_buffer('static_initialized', torch.tensor(False))
    
    def update(
        self,
        features: torch.Tensor,
        location: torch.Tensor,
        scale: torch.Tensor,
        timestamp: int,
        confidence: float = 1.0,
    ):
        """
        Update memory with new information.
        
        Args:
            features: Feature vector (D,)
            location: Center location (2,) [x, y]
            scale: Object scale (2,) [width, height]
            timestamp: Frame timestamp
            confidence: Confidence score
        """
        ptr = self.write_ptr.item()
        
        # Store information
        self.feature_memory[ptr] = features
        self.location_memory[ptr] = location
        self.scale_memory[ptr] = scale
        self.timestamp_memory[ptr] = timestamp
        self.confidence_memory[ptr] = confidence
        
        # Update pointers
        self.write_ptr = (self.write_ptr + 1) % self.memory_size
        self.count = torch.min(self.count + 1, torch.tensor(self.memory_size))
    
    def init_static_query(
        self,
        query_feature: torch.Tensor,
        query_location: torch.Tensor,
    ):
        """
        Initialize static query memory.
        
        Args:
            query_feature: Query feature (D,)
            query_location: Query location (2,)
        """
        if self.use_static:
            self.static_query_feature = query_feature
            self.static_query_location = query_location
            self.static_initialized = torch.tensor(True)
    
    def get_recent(self, k: int) -> Dict[str, torch.Tensor]:
        """
        Get k most recent entries.
        
        Args:
            k: Number of recent entries
        
        Returns:
            Dictionary with recent data
        """
        k = min(k, self.count.item())
        
        if k == 0:
            return {
                'features': torch.zeros(0, self.feature_dim, device=self.feature_memory.device),
                'locations': torch.zeros(0, 2, device=self.location_memory.device),
                'scales': torch.zeros(0, 2, device=self.scale_memory.device),
                'timestamps': torch.zeros(0, dtype=torch.long, device=self.timestamp_memory.device),
                'confidences': torch.zeros(0, device=self.confidence_memory.device),
            }
        
        # Get indices of recent entries
        ptr = self.write_ptr.item()
        if ptr >= k:
            indices = torch.arange(ptr - k, ptr)
        else:
            indices = torch.cat([
                torch.arange(self.memory_size - (k - ptr), self.memory_size),
                torch.arange(0, ptr)
            ])
        
        return {
            'features': self.feature_memory[indices],
            'locations': self.location_memory[indices],
            'scales': self.scale_memory[indices],
            'timestamps': self.timestamp_memory[indices],
            'confidences': self.confidence_memory[indices],
        }
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all valid entries."""
        count = self.count.item()
        
        if count == 0:
            return self.get_recent(0)
        
        return {
            'features': self.feature_memory[:count],
            'locations': self.location_memory[:count],
            'scales': self.scale_memory[:count],
            'timestamps': self.timestamp_memory[:count],
            'confidences': self.confidence_memory[:count],
        }
    
    def predict_motion(
        self,
        current_timestamp: int,
        window: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next location and scale based on motion history.
        
        Args:
            current_timestamp: Current timestamp
            window: Number of frames to use for prediction
        
        Returns:
            - Predicted location (2,)
            - Predicted scale (2,)
        """
        recent = self.get_recent(window)
        
        if len(recent['locations']) < 2:
            # Not enough history, return last known
            if self.count > 0:
                return self.location_memory[self.write_ptr - 1], self.scale_memory[self.write_ptr - 1]
            else:
                return torch.zeros(2, device=self.location_memory.device), torch.ones(2, device=self.scale_memory.device)
        
        # Linear extrapolation
        locations = recent['locations']
        scales = recent['scales']
        timestamps = recent['timestamps'].float()
        
        # Fit linear model
        # location(t) = a * t + b
        t_mean = timestamps.mean()
        t_centered = timestamps - t_mean
        
        # Predict location
        loc_mean = locations.mean(dim=0)
        cov_t_loc = (t_centered.unsqueeze(1) * (locations - loc_mean)).sum(dim=0)
        var_t = (t_centered ** 2).sum()
        
        if var_t > 0:
            slope_loc = cov_t_loc / var_t
            predicted_location = loc_mean + slope_loc * (current_timestamp - t_mean)
        else:
            predicted_location = loc_mean
        
        # Predict scale (assume constant or slow change)
        predicted_scale = scales.mean(dim=0)
        
        return predicted_location, predicted_scale
    
    def reset(self):
        """Reset memory."""
        self.feature_memory.zero_()
        self.location_memory.zero_()
        self.scale_memory.zero_()
        self.timestamp_memory.zero_()
        self.confidence_memory.zero_()
        self.write_ptr.zero_()
        self.count.zero_()
        
        if self.use_static:
            self.static_query_feature.zero_()
            self.static_query_location.zero_()
            self.static_initialized = torch.tensor(False)


class SpatialMemoryAggregator(nn.Module):
    """
    Aggregates spatial information from memory bank.
    Generates spatial prior for next frame prediction.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature aggregation network
        self.feature_aggregator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Location/scale processing
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # [x, y, w, h]
            nn.ReLU(inplace=True),
        )
        
        # Attention for weighting memory entries
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        memory_data: Dict[str, torch.Tensor],
        query_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate memory information.
        
        Args:
            memory_data: Dictionary from memory bank
            query_feature: Optional query feature for attention
        
        Returns:
            Aggregated feature (D,)
        """
        features = memory_data['features']  # (N, D)
        locations = memory_data['locations']  # (N, 2)
        scales = memory_data['scales']  # (N, 2)
        confidences = memory_data['confidences']  # (N,)
        
        N = features.shape[0]
        
        if N == 0:
            return torch.zeros(self.feature_dim, device=features.device)
        
        # Process features
        feat_encoded = self.feature_aggregator(features)  # (N, H)
        
        # Process spatial information
        spatial_info = torch.cat([locations, scales], dim=1)  # (N, 4)
        spatial_encoded = self.spatial_encoder(spatial_info)  # (N, H)
        
        # Compute attention weights
        combined = torch.cat([feat_encoded, spatial_encoded], dim=1)  # (N, 2H)
        
        if query_feature is not None:
            # Use query for attention
            query_encoded = self.feature_aggregator(query_feature.unsqueeze(0))  # (1, H)
            query_broadcast = query_encoded.expand(N, -1)  # (N, H)
            
            attention_input = torch.cat([
                combined,
                torch.cat([query_broadcast, spatial_encoded], dim=1)
            ], dim=1)
        else:
            attention_input = combined
        
        # Compute attention
        attention_weights = self.attention(attention_input)  # (N, 1)
        
        # Weight by confidence
        attention_weights = attention_weights * confidences.unsqueeze(1)
        attention_weights = attention_weights / (attention_weights.sum() + 1e-8)
        
        # Aggregate
        aggregated = (combined * attention_weights).sum(dim=0)  # (2H,)
        
        # Project to output
        output = self.output_proj(aggregated)  # (D,)
        
        return output