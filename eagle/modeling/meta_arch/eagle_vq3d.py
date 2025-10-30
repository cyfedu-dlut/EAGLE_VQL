"""
EAGLE-VQ3D: Complete model for 3D Visual Query tasks.
Extends VQ2D with temporal modeling and 3D understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .eagle_vq2d import EAGLE_VQ2D
from ..temporal import TemporalAggregator, MotionPredictor

logger = logging.getLogger(__name__)


class EAGLE_VQ3D(EAGLE_VQ2D):
    """
    EAGLE model for VQ3D task.
    
    Extends VQ2D with:
    1. Temporal aggregation across frames
    2. Motion prediction
    3. 3D spatial reasoning
    4. Long-term memory
    
    Args:
        cfg: Configuration object
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Additional components for 3D
        logger.info("Building temporal modules for VQ3D...")
        
        # Temporal aggregator
        self.temporal_aggregator = TemporalAggregator(
            feature_dim=cfg.MODEL.TEMPORAL.FEATURE_DIM,
            num_frames=cfg.MODEL.TEMPORAL.NUM_FRAMES,
            aggregation_type=cfg.MODEL.TEMPORAL.AGGREGATION_TYPE,
        )
        
        # Motion predictor
        self.motion_predictor = MotionPredictor(
            feature_dim=cfg.MODEL.TEMPORAL.FEATURE_DIM,
            hidden_dim=cfg.MODEL.TEMPORAL.HIDDEN_DIM,
            prediction_horizon=cfg.MODEL.TEMPORAL.PREDICTION_HORIZON,
        )
        
        # 3D spatial reasoning
        if cfg.MODEL.VQ3D.USE_3D_REASONING:
            self.spatial_3d = Spatial3DReasoning(
                feature_dim=cfg.MODEL.TEMPORAL.FEATURE_DIM,
            )
        else:
            self.spatial_3d = None
        
        # Long-term memory
        self.long_term_memory_size = cfg.MODEL.VQ3D.LONG_TERM_MEMORY_SIZE
        self.use_long_term_memory = cfg.MODEL.VQ3D.USE_LONG_TERM_MEMORY
        
        logger.info("EAGLE-VQ3D model built successfully!")
    
    def forward(
        self,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        search_images: torch.Tensor,
        search_masks: Optional[torch.Tensor] = None,
        frame_indices: Optional[List[int]] = None,
        camera_info: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VQ3D.
        
        Args:
            query_images: Query images (B, 3, H, W)
            query_masks: Query masks (B, 1, H, W)
            search_images: Search video frames (B, T, 3, H, W)
            search_masks: Ground truth masks (B, T, 1, H, W), optional
            frame_indices: Frame indices in video
            camera_info: Camera parameters (intrinsics, extrinsics), optional
        
        Returns:
            Dictionary with predictions and auxiliary outputs
        """
        B, T, C, H, W = search_images.shape
        
        if frame_indices is None:
            frame_indices = list(range(T))
        
        # Extract query features
        query_features = self._extract_features(query_images)
        
        # Extract search features with temporal modeling
        search_features_list = []
        for t in range(T):
            search_feat_t = self._extract_features(search_images[:, t])
            search_features_list.append(search_feat_t)
        search_features = torch.stack(search_features_list, dim=1)  # (B, T, D, H', W')
        
        # Temporal aggregation
        aggregated_features = self.temporal_aggregator(search_features)
        
        # AMM branch with temporal features
        amm_output = self.amm_head(
            query_features=query_features,
            query_mask=F.interpolate(
                query_masks,
                size=query_features.shape[2:],
                mode='bilinear',
                align_corners=False
            ),
            search_features=aggregated_features,
            update_memory=self.training,
        )
        
        amm_predictions = amm_output['predictions']
        
        # GLM branch with temporal tracking
        glm_output = self.glm_head(
            query_features=query_features,
            query_mask=F.interpolate(
                query_masks,
                size=query_features.shape[2:],
                mode='bilinear',
                align_corners=False
            ),
            search_features=search_features,
            update_memory=self.training,
            frame_indices=frame_indices,
        )
        
        glm_predictions = glm_output['predictions']
        
        # Motion prediction
        motion_output = self.motion_predictor(
            features=search_features,
            predictions=glm_predictions,
            frame_indices=frame_indices,
        )
        
        # 3D spatial reasoning (if enabled)
        if self.spatial_3d is not None and camera_info is not None:
            spatial_3d_output = self.spatial_3d(
                features=search_features,
                predictions=glm_predictions,
                camera_info=camera_info,
            )
            
            # Enhance predictions with 3D information
            glm_predictions = glm_predictions + spatial_3d_output['spatial_prior']
            glm_predictions = torch.clamp(glm_predictions, 0.0, 1.0)
        
        # Fusion with temporal consistency
        fusion_output = self.fusion_decoder(
            amm_prediction=amm_predictions,
            glm_prediction=glm_predictions,
            amm_features=None,
            glm_features=None,
            search_features=aggregated_features,
        )
        
        fused_predictions = fusion_output['predictions']
        
        # Apply temporal smoothing
        fused_predictions = self._temporal_smoothing(fused_predictions)
        
        # Upsample to original resolution
        final_predictions = F.interpolate(
            fused_predictions.reshape(B * T, 1, *fused_predictions.shape[-2:]),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).reshape(B, T, 1, H, W)
        
        # Prepare output
        output = {
            'predictions': final_predictions,
            'amm_predictions': F.interpolate(
                amm_predictions.reshape(B * T, 1, *amm_predictions.shape[-2:]),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).reshape(B, T, 1, H, W),
            'glm_predictions': F.interpolate(
                glm_predictions.reshape(B * T, 1, *glm_predictions.shape[-2:]),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).reshape(B, T, 1, H, W),
            'fused_predictions': F.interpolate(
                fused_predictions.reshape(B * T, 1, *fused_predictions.shape[-2:]),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).reshape(B, T, 1, H, W),
            'motion_predictions': motion_output.get('predicted_motion'),
            'temporal_features': aggregated_features,
        }
        
        # Add auxiliary outputs
        if 'intermediate_predictions' in amm_output:
            output['amm_intermediate'] = amm_output['intermediate_predictions']
        if 'locations' in glm_output:
            output['glm_locations'] = glm_output['locations']
            output['glm_scales'] = glm_output['scales']
        if 'predicted_locations' in motion_output:
            output['predicted_locations'] = motion_output['predicted_locations']
        if self.spatial_3d is not None and camera_info is not None:
            output['spatial_3d_output'] = spatial_3d_output
        
        # Compute losses if training
        if self.training and search_masks is not None:
            losses = self._compute_losses_vq3d(output, search_masks, motion_output)
            output['losses'] = losses
        
        return output
    
    def _temporal_smoothing(
        self,
        predictions: torch.Tensor,
        kernel_size: int = 3,
    ) -> torch.Tensor:
        """
        Apply temporal smoothing to predictions.
        
        Args:
            predictions: Predictions (B, T, 1, H, W)
            kernel_size: Temporal kernel size
        
        Returns:
            Smoothed predictions (B, T, 1, H, W)
        """
        B, T, C, H, W = predictions.shape
        
        if T < kernel_size:
            return predictions
        
        # Reshape for temporal convolution
        predictions_reshaped = predictions.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        predictions_reshaped = predictions_reshaped.reshape(B * C, T, H, W).unsqueeze(1)  # (B*C, 1, T, H, W)
        
        # Create temporal smoothing kernel
        kernel = torch.ones(1, 1, kernel_size, 1, 1, device=predictions.device) / kernel_size
        
        # Apply convolution
        padding = kernel_size // 2
        smoothed = F.conv3d(
            predictions_reshaped,
            kernel,
            padding=(padding, 0, 0)
        )
        
        # Reshape back
        smoothed = smoothed.squeeze(1).reshape(B, C, T, H, W)
        smoothed = smoothed.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        
        return smoothed
    
    def _compute_losses_vq3d(
        self,
        output: Dict[str, torch.Tensor],
        target_masks: torch.Tensor,
        motion_output: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for VQ3D including temporal consistency.
        
        Args:
            output: Model output dictionary
            target_masks: Ground truth masks (B, T, 1, H, W)
            motion_output: Motion prediction output
        
        Returns:
            Dictionary of losses
        """
        # Base losses from VQ2D
        losses = self._compute_losses(output, target_masks)
        
        # Temporal consistency loss
        if output['predictions'].size(1) > 1:
            temporal_loss = self._temporal_consistency_loss(output['predictions'])
            losses['loss_temporal'] = 0.1 * temporal_loss
            losses['loss_total'] = losses['loss_total'] + losses['loss_temporal']
        
        # Motion prediction loss (if available)
        if 'predicted_locations' in motion_output and 'glm_locations' in output:
            motion_loss = self._motion_prediction_loss(
                motion_output['predicted_locations'],
                output['glm_locations']
            )
            losses['loss_motion'] = 0.05 * motion_loss
            losses['loss_total'] = losses['loss_total'] + losses['loss_motion']
        
        return losses
    
    def _temporal_consistency_loss(
        self,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        Encourages smooth predictions across frames.
        
        Args:
            predictions: Predictions (B, T, 1, H, W)
        
        Returns:
            Loss value
        """
        B, T, C, H, W = predictions.shape
        
        if T < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute differences between consecutive frames
        diffs = predictions[:, 1:] - predictions[:, :-1]
        
        # L2 loss on differences
        loss = (diffs ** 2).mean()
        
        return loss
    
    def _motion_prediction_loss(
        self,
        predicted_locations: torch.Tensor,
        actual_locations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute motion prediction loss.
        
        Args:
            predicted_locations: Predicted locations (B, T, 2)
            actual_locations: Actual locations (B, T, 2)
        
        Returns:
            Loss value
        """
        # L2 distance between predicted and actual locations
        loss = F.mse_loss(predicted_locations, actual_locations)
        return loss


class Spatial3DReasoning(nn.Module):
    """
    3D Spatial Reasoning module.
    
    Uses camera parameters to perform 3D reasoning:
    1. Depth estimation
    2. 3D position estimation
    3. Spatial constraints
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Depth estimation network
        self.depth_estimator = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )
        
        # 3D position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        
        # Spatial prior generator
        self.spatial_prior_net = nn.Sequential(
            nn.Conv2d(feature_dim + 256 + 1, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        camera_info: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform 3D spatial reasoning.
        
        Args:
            features: Features (B, T, D, H, W)
            predictions: Current predictions (B, T, 1, H, W)
            camera_info: Camera parameters dict with 'intrinsics' and 'extrinsics'
        
        Returns:
            Dictionary with 3D reasoning outputs
        """
        B, T, D, H, W = features.shape
        
        # Estimate depth
        depth_maps = []
        for t in range(T):
            depth_t = self.depth_estimator(features[:, t])
            depth_maps.append(depth_t)
        depth_maps = torch.stack(depth_maps, dim=1)  # (B, T, 1, H, W)
        
        # Back-project to 3D
        points_3d = self._backproject_to_3d(
            predictions,
            depth_maps,
            camera_info['intrinsics']
        )
        
        # Encode 3D positions
        position_features = self._encode_3d_positions(points_3d, features.shape)
        
        # Generate spatial prior
        spatial_priors = []
        for t in range(T):
            prior_input = torch.cat([
                features[:, t],
                position_features[:, t],
                depth_maps[:, t],
            ], dim=1)
            
            spatial_prior = self.spatial_prior_net(prior_input)
            spatial_priors.append(spatial_prior)
        
        spatial_priors = torch.stack(spatial_priors, dim=1)
        
        return {
            'depth_maps': depth_maps,
            'points_3d': points_3d,
            'spatial_prior': spatial_priors,
        }
    
    def _backproject_to_3d(
        self,
        predictions: torch.Tensor,
        depth_maps: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Back-project 2D predictions to 3D using depth.
        
        Args:
            predictions: 2D predictions (B, T, 1, H, W)
            depth_maps: Depth maps (B, T, 1, H, W)
            intrinsics: Camera intrinsics (B, 3, 3)
        
        Returns:
            3D points (B, T, 3, H, W)
        """
        B, T, _, H, W = predictions.shape
        
        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=predictions.device, dtype=torch.float32),
            torch.arange(W, device=predictions.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Normalize to [-1, 1]
        x_coords = (x_coords / W) * 2 - 1
        y_coords = (y_coords / H) * 2 - 1
        
        # Back-project
        points_3d_list = []
        for t in range(T):
            depth_t = depth_maps[:, t, 0]  # (B, H, W)
            
            # Simple back-projection (can be improved with actual intrinsics)
            x_3d = x_coords.unsqueeze(0) * depth_t
            y_3d = y_coords.unsqueeze(0) * depth_t
            z_3d = depth_t
            
            points_3d_t = torch.stack([x_3d, y_3d, z_3d], dim=1)  # (B, 3, H, W)
            points_3d_list.append(points_3d_t)
        
        points_3d = torch.stack(points_3d_list, dim=1)  # (B, T, 3, H, W)
        
        return points_3d
    
    def _encode_3d_positions(
        self,
        points_3d: torch.Tensor,
        target_shape: Tuple,
    ) -> torch.Tensor:
        """
        Encode 3D positions to feature space.
        
        Args:
            points_3d: 3D points (B, T, 3, H, W)
            target_shape: Target shape for output (B, T, D, H, W)
        
        Returns:
            Position features (B, T, 256, H, W)
        """
        B, T, _, H, W = points_3d.shape
        
        # Reshape for encoding
        points_flat = points_3d.permute(0, 1, 3, 4, 2).reshape(B * T * H * W, 3)
        
        # Encode
        encoded = self.position_encoder(points_flat)  # (B*T*H*W, 256)
        
        # Reshape back
        encoded = encoded.reshape(B, T, H, W, 256).permute(0, 1, 4, 2, 3)
        
        return encoded


class TemporalAggregator(nn.Module):
    """
    Temporal feature aggregation across video frames.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_frames: int = 8,
        aggregation_type: str = 'transformer',
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.aggregation_type = aggregation_type
        
        if aggregation_type == 'transformer':
            self.temporal_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=feature_dim,
                    nhead=8,
                    dim_feedforward=feature_dim * 4,
                    dropout=0.1,
                ),
                num_layers=2,
            )
        elif aggregation_type == 'lstm':
            self.temporal_lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=feature_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.lstm_proj = nn.Linear(feature_dim * 2, feature_dim)
        elif aggregation_type == 'attention':
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                dropout=0.1,
            )
        else:
            # Simple averaging
            pass
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal features.
        
        Args:
            features: Features (B, T, D, H, W)
        
        Returns:
            Aggregated features (B, T, D, H, W)
        """
        B, T, D, H, W = features.shape
        
        if self.aggregation_type == 'mean':
            # Simple averaging
            return features
        
        # Reshape for temporal processing
        features_flat = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, D)
        
        if self.aggregation_type == 'transformer':
            # Transformer encoding
            aggregated = self.temporal_transformer(features_flat.permute(1, 0, 2))
            aggregated = aggregated.permute(1, 0, 2)  # (B*H*W, T, D)
        
        elif self.aggregation_type == 'lstm':
            # LSTM encoding
            aggregated, _ = self.temporal_lstm(features_flat)
            aggregated = self.lstm_proj(aggregated)
        
        elif self.aggregation_type == 'attention':
            # Self-attention
            aggregated, _ = self.temporal_attention(
                features_flat.permute(1, 0, 2),
                features_flat.permute(1, 0, 2),
                features_flat.permute(1, 0, 2),
            )
            aggregated = aggregated.permute(1, 0, 2)
        
        # Reshape back
        aggregated = aggregated.reshape(B, H, W, T, D).permute(0, 3, 4, 1, 2)
        
        return aggregated


class MotionPredictor(nn.Module):
    """
    Motion prediction module for anticipating future object positions.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        prediction_horizon: int = 5,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        
        # Motion encoder
        self.motion_encoder = nn.LSTM(
            input_size=2,  # (x, y) positions
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        
        # Motion predictor
        self.motion_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, prediction_horizon * 2),  # (x, y) for each future frame
        )
    
    def forward(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        frame_indices: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future motion.
        
        Args:
            features: Features (B, T, D, H, W)
            predictions: Current predictions (B, T, 1, H, W)
            frame_indices: Frame indices
        
        Returns:
            Dictionary with motion predictions
        """
        B, T = features.shape[:2]
        
        # Extract locations from predictions
        locations = self._extract_locations(predictions)  # (B, T, 2)
        
        # Encode motion history
        motion_features, (h_n, c_n) = self.motion_encoder(locations)
        
        # Predict future motion
        last_hidden = h_n[-1]  # (B, H)
        predicted_motion = self.motion_predictor(last_hidden)  # (B, horizon * 2)
        predicted_motion = predicted_motion.reshape(B, self.prediction_horizon, 2)
        
        # Compute predicted locations (relative to last known)
        last_location = locations[:, -1:, :]  # (B, 1, 2)
        predicted_locations = last_location + predicted_motion.cumsum(dim=1)
        
        return {
            'predicted_motion': predicted_motion,
            'predicted_locations': predicted_locations,
            'motion_features': motion_features,
        }
    
    def _extract_locations(self, predictions: torch.Tensor) -> torch.Tensor:
        """Extract center locations from predictions."""
        B, T, _, H, W = predictions.shape
        
        locations = []
        
        for b in range(B):
            for t in range(T):
                pred = predictions[b, t, 0]
                
                # Compute center of mass
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(H, device=pred.device, dtype=torch.float32),
                    torch.arange(W, device=pred.device, dtype=torch.float32),
                    indexing='ij'
                )
                
                pred_sum = pred.sum().clamp(min=1e-8)
                center_y = (pred * y_coords).sum() / pred_sum
                center_x = (pred * x_coords).sum() / pred_sum
                
                locations.append(torch.stack([center_x, center_y]))
        
        locations = torch.stack(locations).reshape(B, T, 2)
        
        return locations