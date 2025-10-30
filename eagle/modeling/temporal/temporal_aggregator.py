"""
Temporal Aggregator for combining features across time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalAggregator(nn.Module):
    """
    Temporal Feature Aggregator.
    
    Combines features across temporal dimension using various strategies:
    - Mean pooling
    - Max pooling
    - LSTM
    - Transformer
    - Attention
    
    Args:
        feature_dim: Feature dimension
        num_frames: Number of frames to aggregate
        aggregation_type: Type of aggregation
        return_all_frames: Whether to return features for all frames
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_frames: int = 8,
        aggregation_type: str = 'transformer',
        return_all_frames: bool = True,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.aggregation_type = aggregation_type
        self.return_all_frames = return_all_frames
        
        # Build aggregation module
        if aggregation_type == 'transformer':
            self.aggregator = TemporalTransformer(
                feature_dim=feature_dim,
                num_heads=8,
                num_layers=2,
            )
        
        elif aggregation_type == 'lstm':
            self.aggregator = TemporalLSTM(
                feature_dim=feature_dim,
                hidden_dim=feature_dim,
                num_layers=2,
                bidirectional=True,
            )
        
        elif aggregation_type == 'attention':
            self.aggregator = TemporalSelfAttention(
                feature_dim=feature_dim,
                num_heads=8,
            )
        
        elif aggregation_type == 'gru':
            self.aggregator = TemporalGRU(
                feature_dim=feature_dim,
                hidden_dim=feature_dim,
                num_layers=2,
                bidirectional=True,
            )
        
        elif aggregation_type in ['mean', 'max', 'sum']:
            self.aggregator = None  # Use simple operations
        
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate temporal features.
        
        Args:
            features: Features (B, T, D, H, W)
            mask: Optional mask for valid frames (B, T)
        
        Returns:
            Aggregated features (B, T, D, H, W) or (B, D, H, W)
        """
        B, T, D, H, W = features.shape
        
        # Simple aggregation
        if self.aggregation_type == 'mean':
            if mask is not None:
                mask_expanded = mask.view(B, T, 1, 1, 1).expand_as(features)
                aggregated = (features * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True).view(B, 1, 1, 1)
            else:
                aggregated = features.mean(dim=1)
            
            if self.return_all_frames:
                aggregated = aggregated.unsqueeze(1).expand(-1, T, -1, -1, -1)
            
            return aggregated
        
        elif self.aggregation_type == 'max':
            aggregated = features.max(dim=1)[0]
            
            if self.return_all_frames:
                aggregated = aggregated.unsqueeze(1).expand(-1, T, -1, -1, -1)
            
            return aggregated
        
        elif self.aggregation_type == 'sum':
            aggregated = features.sum(dim=1)
            
            if self.return_all_frames:
                aggregated = aggregated.unsqueeze(1).expand(-1, T, -1, -1, -1)
            
            return aggregated
        
        # Complex aggregation with learnable modules
        # Reshape for temporal processing: (B*H*W, T, D)
        features_reshaped = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
        
        # Apply aggregator
        aggregated = self.aggregator(features_reshaped, mask)  # (B*H*W, T, D) or (B*H*W, D)
        
        # Reshape back
        if aggregated.dim() == 2:
            # Single frame output
            aggregated = aggregated.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
            
            if self.return_all_frames:
                aggregated = aggregated.unsqueeze(1).expand(-1, T, -1, -1, -1)
        else:
            # Multi-frame output
            aggregated = aggregated.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2)  # (B, T, D, H, W)
        
        return aggregated


class TemporalTransformer(nn.Module):
    """Transformer-based temporal aggregation."""
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply transformer.
        
        Args:
            features: Features (N, T, D)
            mask: Mask (B, T) - needs to be expanded to (N, T)
        
        Returns:
            Aggregated features (N, T, D)
        """
        # Transformer expects batch_first=True
        output = self.transformer(features)
        return output


class TemporalLSTM(nn.Module):
    """LSTM-based temporal aggregation."""
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Project back to feature_dim if needed
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if output_dim != feature_dim:
            self.output_proj = nn.Linear(output_dim, feature_dim)
        else:
            self.output_proj = None
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply LSTM.
        
        Args:
            features: Features (N, T, D)
            mask: Mask (B, T)
        
        Returns:
            Aggregated features (N, T, D)
        """
        output, (h_n, c_n) = self.lstm(features)
        
        if self.output_proj is not None:
            output = self.output_proj(output)
        
        return output


class TemporalGRU(nn.Module):
    """GRU-based temporal aggregation."""
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Project back to feature_dim if needed
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if output_dim != feature_dim:
            self.output_proj = nn.Linear(output_dim, feature_dim)
        else:
            self.output_proj = None
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply GRU.
        
        Args:
            features: Features (N, T, D)
            mask: Mask (B, T)
        
        Returns:
            Aggregated features (N, T, D)
        """
        output, h_n = self.gru(features)
        
        if self.output_proj is not None:
            output = self.output_proj(output)
        
        return output


class TemporalSelfAttention(nn.Module):
    """Self-attention based temporal aggregation."""
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply self-attention.
        
        Args:
            features: Features (N, T, D)
            mask: Mask (B, T)
        
        Returns:
            Aggregated features (N, T, D)
        """
        # Self-attention
        attn_output, _ = self.attention(features, features, features)
        
        # Residual connection
        output = self.norm(features + attn_output)
        
        return output


class AdaptiveTemporalAggregator(nn.Module):
    """
    Adaptive temporal aggregator that learns to weight different frames.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_frames: int = 8,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # Frame importance predictor
        self.importance_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Adaptively aggregate features.
        
        Args:
            features: Features (N, T, D)
            mask: Mask (B, T)
        
        Returns:
            Aggregated features (N, T, D)
        """
        N, T, D = features.shape
        
        # Compute importance scores for each frame
        importance = self.importance_net(features)  # (N, T, 1)
        importance = torch.softmax(importance, dim=1)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match features
            # mask is (B, T), but features is (N, T, D) where N = B*H*W
            # This is a simplification; in practice you'd need to handle this properly
            importance = importance * mask.unsqueeze(-1)
            importance = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted aggregation
        aggregated = (features * importance).sum(dim=1, keepdim=True)  # (N, 1, D)
        aggregated = aggregated.expand(-1, T, -1)  # (N, T, D)
        
        # Refine
        output = self.fusion(aggregated)
        
        return output