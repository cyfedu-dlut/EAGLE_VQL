"""
Visualization utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional
import cv2


def visualize_predictions(
    query_image: torch.Tensor,
    query_mask: torch.Tensor,
    search_image: torch.Tensor,
    prediction: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Visualize query, search, prediction, and target.
    
    Args:
        query_image: Query image (3, H, W)
        query_mask: Query mask (1, H, W)
        search_image: Search image(s) (3, H, W) or (T, 3, H, W)
        prediction: Prediction(s) (1, H, W) or (T, 1, H, W)
        target: Target(s) (1, H, W) or (T, 1, H, W), optional
        save_path: Path to save visualization
        show: Whether to show visualization
    """
    # Convert to numpy
    query_image = tensor_to_image(query_image)
    query_mask = tensor_to_mask(query_mask)
    
    # Handle multi-frame case
    if search_image.dim() == 4:
        # Visualize first frame only
        search_image = search_image[0]
        prediction = prediction[0]
        if target is not None:
            target = target[0]
    
    search_image = tensor_to_image(search_image)
    prediction = tensor_to_mask(prediction)
    
    if target is not None:
        target = tensor_to_mask(target)
    
    # Create figure
    n_cols = 4 if target is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Query image with mask overlay
    axes[0].imshow(query_image)
    axes[0].imshow(query_mask, alpha=0.5, cmap='jet')
    axes[0].set_title('Query')
    axes[0].axis('off')
    
    # Search image
    axes[1].imshow(search_image)
    axes[1].set_title('Search')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(search_image)
    axes[2].imshow(prediction, alpha=0.5, cmap='jet')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Target (if available)
    if target is not None:
        axes[3].imshow(search_image)
        axes[3].imshow(target, alpha=0.5, cmap='jet')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Show
    if show:
        plt.show()
    
    plt.close()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image.
    
    Args:
        tensor: Image tensor (3, H, W)
    
    Returns:
        Numpy image (H, W, 3) in range [0, 1]
    """
    image = tensor.cpu().numpy()
    
    # Denormalize if needed (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = image * std + mean
    
    # Transpose to (H, W, 3)
    image = np.transpose(image, (1, 2, 0))
    
    # Clip to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def tensor_to_mask(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy mask.
    
    Args:
        tensor: Mask tensor (1, H, W)
    
    Returns:
        Numpy mask (H, W)
    """
    mask = tensor.cpu().numpy()
    
    if mask.ndim == 3:
        mask = mask[0]
    
    return mask


def save_video_predictions(
    search_images: torch.Tensor,
    predictions: torch.Tensor,
    save_path: str,
    fps: int = 10,
):
    """
    Save video predictions as video file.
    
    Args:
        search_images: Search images (T, 3, H, W)
        predictions: Predictions (T, 1, H, W)
        save_path: Path to save video
        fps: Frames per second
    """
    T = search_images.shape[0]
    H, W = search_images.shape[2:]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    
    for t in range(T):
        # Get frame
        image = tensor_to_image(search_images[t])
        prediction = tensor_to_mask(predictions[t])
        
        # Overlay prediction
        overlay = (image * 255).astype(np.uint8)
        mask_colored = (plt.cm.jet(prediction)[:, :, :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Convert RGB to BGR for OpenCV
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(overlay)
    
    out.release()


def visualize_attention(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Visualize attention weights.
    
    Args:
        attention_weights: Attention weights (H, T_q, T_k)
        save_path: Path to save visualization
    """
    # Average over heads
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.mean(dim=1)
    
    attention_weights = attention_weights.cpu().numpy()
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[0], cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Key positions')
    plt.ylabel('Query positions')
    plt.title('Attention Weights')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close()