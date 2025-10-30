"""
Spatial Decoder for upsampling and refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDecoder(nn.Module):
    """
    Spatial Decoder for upsampling predictions to original resolution.
    
    Uses progressive upsampling with skip connections.
    
    Args:
        feature_dim: Input feature dimension
        output_stride: Output stride relative to input (e.g., 4, 8, 16)
        use_skip_connections: Use skip connections from encoder
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        output_stride: int = 16,
        use_skip_connections: bool = True,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.output_stride = output_stride
        self.use_skip_connections = use_skip_connections
        
        # Calculate number of upsampling stages
        self.num_stages = int(torch.log2(torch.tensor(output_stride)).item())
        
        # Build upsampling stages
        self.upsample_stages = nn.ModuleList()
        
        current_dim = feature_dim
        for i in range(self.num_stages):
            stage = nn.Sequential(
                nn.ConvTranspose2d(
                    current_dim,
                    current_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(current_dim // 2),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(current_dim // 2, current_dim // 2, 3, padding=1),
                nn.BatchNorm2d(current_dim // 2),
                nn.ReLU(inplace=True),
            )
            
            self.upsample_stages.append(stage)
            current_dim = current_dim // 2
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(current_dim, 1, 1)
    
    def forward(
        self,
        features: torch.Tensor,
        skip_features: list = None,
    ) -> torch.Tensor:
        """
        Upsample features to full resolution.
        
        Args:
            features: Low-resolution features (B, D, H, W)
            skip_features: List of skip connection features from encoder
        
        Returns:
            High-resolution prediction (B, 1, H', W')
        """
        x = features
        
        for i, stage in enumerate(self.upsample_stages):
            # Upsample
            x = stage(x)
            
            # Add skip connection if available
            if self.use_skip_connections and skip_features is not None:
                if i < len(skip_features):
                    skip = skip_features[-(i+1)]  # Reverse order
                    
                    # Match dimensions
                    if skip.shape[2:] != x.shape[2:]:
                        skip = F.interpolate(
                            skip,
                            size=x.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    if skip.shape[1] != x.shape[1]:
                        # Project skip connection to match channels
                        skip = F.conv2d(
                            skip,
                            torch.randn(x.shape[1], skip.shape[1], 1, 1, device=x.device) * 0.01,
                        )
                    
                    x = x + skip
        
        # Final prediction
        output = self.final_conv(x)
        output = torch.sigmoid(output)
        
        return output


class ASPPDecoder(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) Decoder.
    Captures multi-scale context for better segmentation.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        atrous_rates: list = [6, 12, 18],
        output_dim: int = 256,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.atrous_rates = atrous_rates
        self.output_dim = output_dim
        
        # ASPP branches
        self.aspp_branches = nn.ModuleList()
        
        # 1x1 convolution
        self.aspp_branches.append(
            nn.Sequential(
                nn.Conv2d(feature_dim, output_dim, 1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        )
        
        # Atrous convolutions
        for rate in atrous_rates:
            self.aspp_branches.append(
                nn.Sequential(
                    nn.Conv2d(feature_dim, output_dim, 3, padding=rate, dilation=rate),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True),
                )
            )
        
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, output_dim, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(output_dim * (len(atrous_rates) + 2), output_dim, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        
        # Final prediction
        self.prediction = nn.Conv2d(output_dim, 1, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply ASPP and generate prediction.
        
        Args:
            features: Input features (B, D, H, W)
        
        Returns:
            Prediction (B, 1, H, W)
        """
        B, D, H, W = features.shape
        
        # Apply ASPP branches
        aspp_outputs = []
        
        for branch in self.aspp_branches:
            aspp_outputs.append(branch(features))
        
        # Global pooling
        global_feat = self.global_pool(features)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        aspp_outputs.append(global_feat)
        
        # Concatenate and fuse
        concat_features = torch.cat(aspp_outputs, dim=1)
        fused = self.fusion(concat_features)
        
        # Generate prediction
        prediction = self.prediction(fused)
        prediction = torch.sigmoid(prediction)
        
        return prediction