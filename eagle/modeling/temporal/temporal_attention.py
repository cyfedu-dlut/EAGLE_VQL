"""
Temporal Attention mechanisms for video understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TemporalAttention(nn.Module):
    """
    Temporal attention module for computing frame importance.
    
    Args:
        feature_dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temporal attention.
        
        Args:
            query: Query tensor (B, T_q, D)
            key: Key tensor (B, T_k, D)
            value: Value tensor (B, T_v, D)
            mask: Attention mask (B, T_q, T_k)
        
        Returns:
            - Output tensor (B, T_q, D)
            - Attention weights (B, num_heads, T_q, T_k)
        """
        B, T_q, D = query.shape
        T_k = key.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # (B, T_q, D)
        K = self.k_proj(key)    # (B, T_k, D)
        V = self.v_proj(value)  # (B, T_v, D)
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_q, d)
        K = K.reshape(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_k, d)
        V = V.reshape(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_v, d)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T_q, T_k)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, T_q, T_k)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, T_q, T_k)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # (B, H, T_q, d)
        
        # Reshape and project
        output = output.transpose(1, 2).reshape(B, T_q, D)  # (B, T_q, D)
        output = self.out_proj(output)
        
        return output, attn_weights


class SpatioTemporalAttention(nn.Module):
    """
    Spatio-temporal attention that attends over both space and time.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Temporal attention
        self.temporal_attn = TemporalAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Spatial attention
        self.spatial_attn = TemporalAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply spatio-temporal attention.
        
        Args:
            features: Features (B, T, H*W, D)
            temporal_mask: Temporal attention mask
            spatial_mask: Spatial attention mask
        
        Returns:
            Attended features (B, T, H*W, D)
        """
        B, T, N, D = features.shape  # N = H * W
        
        # Temporal attention (across time for each spatial location)
        features_reshaped = features.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
        
        temporal_out, _ = self.temporal_attn(
            features_reshaped,
            features_reshaped,
            features_reshaped,
            mask=temporal_mask,
        )
        
        temporal_out = temporal_out.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        features = self.norm1(features + temporal_out)
        
        # Spatial attention (across space for each time step)
        features_reshaped = features.reshape(B * T, N, D)  # (B*T, N, D)
        
        spatial_out, _ = self.spatial_attn(
            features_reshaped,
            features_reshaped,
            features_reshaped,
            mask=spatial_mask,
        )
        
        spatial_out = spatial_out.reshape(B, T, N, D)  # (B, T, N, D)
        features = self.norm2(features + spatial_out)
        
        # FFN
        ffn_out = self.ffn(features)
        features = self.norm3(features + ffn_out)
        
        return features


class DividedSpaceTimeAttention(nn.Module):
    """
    Divided Space-Time Attention (from TimeSformer).
    Separately applies temporal and spatial attention for efficiency.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(feature_dim)
        
        # Spatial attention
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(feature_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply divided space-time attention.
        
        Args:
            x: Features (B, T, N, D) where N = H*W
        
        Returns:
            Attended features (B, T, N, D)
        """
        B, T, N, D = x.shape
        
        # Temporal attention
        # Rearrange to (B*N, T, D)
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        
        attn_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = x_temporal + attn_out
        x_temporal = self.temporal_norm(x_temporal)
        
        # Rearrange back to (B, T, N, D)
        x = x_temporal.reshape(B, N, T, D).permute(0, 2, 1, 3)
        
        # Spatial attention
        # Rearrange to (B*T, N, D)
        x_spatial = x.reshape(B * T, N, D)
        
        attn_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial + attn_out
        x_spatial = self.spatial_norm(x_spatial)
        
        # Rearrange back to (B, T, N, D)
        x = x_spatial.reshape(B, T, N, D)
        
        # FFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ffn_norm(x)
        
        return x


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding for video frames.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        max_len: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, feature_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(10000.0) / feature_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, D)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Features (B, T, D) or (B, T, N, D)
        
        Returns:
            Features with positional encoding
        """
        if x.dim() == 3:
            # (B, T, D)
            x = x + self.pe[:, :x.size(1), :]
        elif x.dim() == 4:
            # (B, T, N, D)
            pe = self.pe[:, :x.size(1), :].unsqueeze(2)  # (1, T, 1, D)
            x = x + pe
        
        return self.dropout(x)


class CrossFrameAttention(nn.Module):
    """
    Cross-frame attention for matching across frames.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = TemporalAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        query_frame: torch.Tensor,
        support_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Match query frame against support frames.
        
        Args:
            query_frame: Query frame features (B, N, D) where N = H*W
            support_frames: Support frame features (B, T, N, D)
        
        Returns:
            Matched features (B, N, D)
        """
        B, T, N, D = support_frames.shape
        
        # Reshape support frames
        support_reshaped = support_frames.reshape(B, T * N, D)  # (B, T*N, D)
        
        # Expand query
        query_expanded = query_frame.unsqueeze(1)  # (B, 1, N, D)
        query_expanded = query_expanded.expand(-1, T, -1, -1).reshape(B, T * N, D)
        
        # Attend
        attended, _ = self.attention(query_frame, support_reshaped, support_reshaped)
        
        # Residual
        output = self.norm(query_frame + attended)
        
        return output