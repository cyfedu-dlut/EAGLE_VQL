"""
EAGLE-VQ2D: Complete model for 2D Visual Query tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from ..backbone import build_backbone
from ..amm import AMMHead
from ..glm import GLMHead
from ..decoder import FusionDecoder, PredictionHead

logger = logging.getLogger(__name__)


class EAGLE_VQ2D(nn.Module):
    """
    EAGLE model for VQ2D task.
    
    Architecture:
    1. Shared DINOv2 backbone
    2. AMM (Appearance-Aware Meta-Learning Memory) branch
    3. GLM (Geometry-Aware Localization Memory) branch
    4. Fusion decoder
    5. Prediction head
    
    Args:
        cfg: Configuration object
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        # Build backbone
        logger.info("Building backbone...")
        self.backbone = build_backbone(cfg)
        
        # Build AMM branch
        logger.info("Building AMM branch...")
        self.amm_head = AMMHead(
            feature_dim=cfg.MODEL.AMM.FEATURE_DIM,
            memory_size=cfg.MODEL.AMM.MEMORY_SIZE,
            pseudo_label_channels=cfg.MODEL.AMM.PSEUDO_LABEL_CHANNELS,
            meta_learner_iters=cfg.MODEL.AMM.META_LEARNER_ITERS,
            learning_rate=cfg.MODEL.AMM.LEARNING_RATE,
            regularizer=cfg.MODEL.AMM.REGULARIZER,
            confidence_threshold=cfg.MODEL.AMM.CONFIDENCE_THRESHOLD,
        )
        
        # Build GLM branch
        logger.info("Building GLM branch...")
        self.glm_head = GLMHead(
            feature_dim=cfg.MODEL.GLM.FEATURE_DIM,
            memory_size=cfg.MODEL.GLM.MEMORY_SIZE,
            dcf_iters=cfg.MODEL.GLM.DCF_ITERS,
            lambda_reg=cfg.MODEL.GLM.LAMBDA_REG,
            use_static=cfg.MODEL.GLM.USE_STATIC,
            scale_factor=cfg.MODEL.GLM.SCALE_FACTOR,
            update_first_n=cfg.MODEL.GLM.UPDATE_FIRST_N,
        )
        
        # Build fusion decoder
        logger.info("Building fusion decoder...")
        self.fusion_decoder = FusionDecoder(
            feature_dim=cfg.MODEL.DECODER.FEATURE_DIM,
            fusion_type=cfg.MODEL.DECODER.FUSION_TYPE,
            num_refinement_stages=cfg.MODEL.DECODER.NUM_REFINEMENT_STAGES,
            use_multi_scale=cfg.MODEL.DECODER.USE_MULTI_SCALE,
        )
        
        # Build prediction head
        self.prediction_head = PredictionHead(
            feature_dim=cfg.MODEL.DECODER.FEATURE_DIM,
            hidden_dim=cfg.MODEL.DECODER.HIDDEN_DIM,
            num_classes=1,
            use_crf=cfg.MODEL.DECODER.USE_CRF,
        )
        
        # Loss weights
        self.amm_weight = cfg.MODEL.LOSS_WEIGHTS.AMM
        self.glm_weight = cfg.MODEL.LOSS_WEIGHTS.GLM
        self.fusion_weight = cfg.MODEL.LOSS_WEIGHTS.FUSION
        
        self.to(self.device)
        
        logger.info("EAGLE-VQ2D model built successfully!")
    
    def forward(
        self,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        search_images: torch.Tensor,
        search_masks: Optional[torch.Tensor] = None,
        frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query_images: Query images (B, 3, H, W)
            query_masks: Query masks (B, 1, H, W)
            search_images: Search images (B, T, 3, H, W) or (B, 3, H, W)
            search_masks: Ground truth masks for training (B, T, 1, H, W), optional
            frame_indices: Frame indices for temporal tracking
        
        Returns:
            Dictionary with predictions and intermediate outputs
        """
        # Handle single frame case
        single_frame = search_images.dim() == 4
        if single_frame:
            search_images = search_images.unsqueeze(1)
            if search_masks is not None:
                search_masks = search_masks.unsqueeze(1)
        
        B, T, C, H, W = search_images.shape
        
        # Extract query features
        query_features = self._extract_features(query_images)
        
        # Extract search features
        search_features_list = []
        for t in range(T):
            search_feat_t = self._extract_features(search_images[:, t])
            search_features_list.append(search_feat_t)
        search_features = torch.stack(search_features_list, dim=1)  # (B, T, D, H', W')
        
        # AMM branch
        amm_output = self.amm_head(
            query_features=query_features,
            query_mask=F.interpolate(
                query_masks, 
                size=query_features.shape[2:], 
                mode='bilinear',
                align_corners=False
            ),
            search_features=search_features,
            update_memory=self.training,
        )
        
        amm_predictions = amm_output['predictions']
        
        # GLM branch
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
        
        # Fusion
        fusion_output = self.fusion_decoder(
            amm_prediction=amm_predictions,
            glm_prediction=glm_predictions,
            amm_features=None,  # Can add if needed
            glm_features=None,
            search_features=search_features,
        )
        
        fused_predictions = fusion_output['predictions']
        
        # Upsample to original resolution
        final_predictions = F.interpolate(
            fused_predictions.reshape(B * T, 1, *fused_predictions.shape[-2:]),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).reshape(B, T, 1, H, W)
        
        if single_frame:
            final_predictions = final_predictions.squeeze(1)
            amm_predictions = amm_predictions.squeeze(1)
            glm_predictions = glm_predictions.squeeze(1)
            fused_predictions = fused_predictions.squeeze(1)
        
        # Prepare output
        output = {
            'predictions': final_predictions,
            'amm_predictions': F.interpolate(
                amm_predictions.reshape(B * T, 1, *amm_predictions.shape[-2:]) if not single_frame else amm_predictions,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).reshape(B, T, 1, H, W) if not single_frame else F.interpolate(amm_predictions, size=(H, W), mode='bilinear', align_corners=False),
            'glm_predictions': F.interpolate(
                glm_predictions.reshape(B * T, 1, *glm_predictions.shape[-2:]) if not single_frame else glm_predictions,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).reshape(B, T, 1, H, W) if not single_frame else F.interpolate(glm_predictions, size=(H, W), mode='bilinear', align_corners=False),
            'fused_predictions': F.interpolate(
                fused_predictions.reshape(B * T, 1, *fused_predictions.shape[-2:]) if not single_frame else fused_predictions,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).reshape(B, T, 1, H, W) if not single_frame else F.interpolate(fused_predictions, size=(H, W), mode='bilinear', align_corners=False),
        }
        
        # Add auxiliary outputs
        if 'intermediate_predictions' in amm_output:
            output['amm_intermediate'] = amm_output['intermediate_predictions']
        if 'pseudo_labels' in amm_output:
            output['pseudo_labels'] = amm_output['pseudo_labels']
        if 'weights' in amm_output:
            output['amm_weights'] = amm_output['weights']
        if 'response_maps' in glm_output:
            output['glm_response_maps'] = glm_output['response_maps']
        if 'locations' in glm_output:
            output['glm_locations'] = glm_output['locations']
        if 'fusion_weights' in fusion_output:
            output['fusion_weights'] = fusion_output['fusion_weights']
        
        # Compute losses if training
        if self.training and search_masks is not None:
            losses = self._compute_losses(output, search_masks)
            output['losses'] = losses
        
        return output
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using backbone.
        
        Args:
            images: Input images (B, 3, H, W)
        
        Returns:
            Features (B, D, H', W')
        """
        backbone_output = self.backbone(images)
        features = backbone_output['features']
        return features
    
    def _compute_losses(
        self,
        output: Dict[str, torch.Tensor],
        target_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            output: Model output dictionary
            target_masks: Ground truth masks (B, T, 1, H, W) or (B, 1, H, W)
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Main prediction loss
        main_loss = self._lovasz_hinge_loss(
            output['predictions'],
            target_masks
        )
        losses['loss_main'] = main_loss
        
        # AMM branch loss
        amm_loss = self._lovasz_hinge_loss(
            output['amm_predictions'],
            target_masks
        )
        losses['loss_amm'] = self.amm_weight * amm_loss
        
        # GLM branch loss
        glm_loss = self._lovasz_hinge_loss(
            output['glm_predictions'],
            target_masks
        )
        losses['loss_glm'] = self.glm_weight * glm_loss
        
        # Fused prediction loss
        fused_loss = self._lovasz_hinge_loss(
            output['fused_predictions'],
            target_masks
        )
        losses['loss_fused'] = self.fusion_weight * fused_loss
        
        # Total loss
        losses['loss_total'] = (
            losses['loss_main'] +
            losses['loss_amm'] +
            losses['loss_glm'] +
            losses['loss_fused']
        )
        
        return losses
    
    def _lovasz_hinge_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lovász hinge loss for binary segmentation.
        
        Args:
            predictions: Predictions (B, 1, H, W) or (B, T, 1, H, W)
            targets: Ground truth (B, 1, H, W) or (B, T, 1, H, W)
        
        Returns:
            Loss value
        """
        # Flatten spatial and temporal dimensions
        if predictions.dim() == 5:
            B, T, C, H, W = predictions.shape
            predictions = predictions.reshape(B * T, C, H, W)
            targets = targets.reshape(B * T, C, H, W)
        
        # Flatten
        predictions_flat = predictions.view(predictions.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        # Compute errors
        errors = torch.abs(predictions_flat - targets_flat)
        
        # Sort errors
        errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
        targets_sorted = targets_flat.gather(1, perm)
        
        # Compute Lovász extension
        losses = []
        for i in range(predictions.size(0)):
            inter = targets_sorted[i].sum() - targets_sorted[i].float().cumsum(0)
            union = targets_sorted[i].sum() + (1 - targets_sorted[i]).float().cumsum(0)
            jaccard = 1.0 - inter / union.clamp(min=1.0)
            
            if len(jaccard) > 1:
                jaccard[1:] = jaccard[1:] - jaccard[:-1]
            
            loss_i = (jaccard * errors_sorted[i]).sum()
            losses.append(loss_i)
        
        return torch.stack(losses).mean()
    
    def reset_memory(self):
        """Reset memory banks for new sequence."""
        self.amm_head.reset_memory()
        self.glm_head.reset()
    
    def inference(
        self,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        search_images: torch.Tensor,
        frame_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Inference mode: return only final predictions.
        
        Args:
            query_images: Query images (B, 3, H, W)
            query_masks: Query masks (B, 1, H, W)
            search_images: Search images (B, T, 3, H, W)
            frame_indices: Frame indices
        
        Returns:
            Predictions (B, T, 1, H, W)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                query_images,
                query_masks,
                search_images,
                frame_indices=frame_indices,
            )
        return output['predictions']


class EAGLE_VQ2D_SAM(EAGLE_VQ2D):
    """
    EAGLE-VQ2D with SAM (Segment Anything Model) integration.
    Uses SAM for initial query mask generation or refinement.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Load SAM if enabled
        if cfg.MODEL.SAM.ENABLED:
            logger.info("Loading SAM model...")
            self.sam = self._load_sam(cfg)
        else:
            self.sam = None
    
    def _load_sam(self, cfg):
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment_anything not installed. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        model_type = cfg.MODEL.SAM.MODEL_TYPE
        checkpoint = cfg.MODEL.SAM.CHECKPOINT
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        
        predictor = SamPredictor(sam)
        
        return predictor
    
    def forward(
        self,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        search_images: torch.Tensor,
        search_masks: Optional[torch.Tensor] = None,
        frame_indices: Optional[List[int]] = None,
        use_sam_refinement: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional SAM refinement.
        
        Args:
            query_images: Query images (B, 3, H, W)
            query_masks: Query masks (B, 1, H, W)
            search_images: Search images (B, T, 3, H, W) or (B, 3, H, W)
            search_masks: Ground truth masks (B, T, 1, H, W), optional
            frame_indices: Frame indices
            use_sam_refinement: Whether to use SAM for refinement
        
        Returns:
            Dictionary with predictions
        """
        # Standard forward pass
        output = super().forward(
            query_images,
            query_masks,
            search_images,
            search_masks,
            frame_indices,
        )
        
        # Refine with SAM if requested
        if use_sam_refinement and self.sam is not None:
            refined_predictions = self._refine_with_sam(
                search_images,
                output['predictions']
            )
            output['predictions_sam_refined'] = refined_predictions
            output['predictions'] = refined_predictions  # Use refined as final
        
        return output
    
    def _refine_with_sam(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine predictions using SAM.
        
        Args:
            images: Input images (B, T, 3, H, W)
            predictions: Initial predictions (B, T, 1, H, W)
        
        Returns:
            Refined predictions (B, T, 1, H, W)
        """
        single_frame = images.dim() == 4
        if single_frame:
            images = images.unsqueeze(1)
            predictions = predictions.unsqueeze(1)
        
        B, T, C, H, W = images.shape
        
        refined_list = []
        
        for b in range(B):
            for t in range(T):
                image = images[b, t].cpu().numpy().transpose(1, 2, 0)
                pred = predictions[b, t, 0].cpu().numpy()
                
                # Set image for SAM
                self.sam.set_image(image)
                
                # Get bbox from prediction
                ys, xs = torch.where(torch.tensor(pred) > 0.5)
                if len(ys) > 0:
                    bbox = [xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()]
                    
                    # Predict with SAM
                    mask_refined, _, _ = self.sam.predict(
                        box=bbox,
                        multimask_output=False,
                    )
                    
                    refined_list.append(torch.from_numpy(mask_refined[0]).float().unsqueeze(0))
                else:
                    refined_list.append(torch.zeros(1, H, W))
        
        refined_predictions = torch.stack(refined_list).reshape(B, T, 1, H, W).to(predictions.device)
        
        if single_frame:
            refined_predictions = refined_predictions.squeeze(1)
        
        return refined_predictions