"""
Inference script for EAGLE models.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eagle.config import get_cfg_defaults
from eagle.modeling.meta_arch import build_model, load_checkpoint
from eagle.data.transforms import build_transforms
from eagle.utils.visualization import visualize_predictions, save_video_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with EAGLE model')
    
    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Path to config file',
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights',
    )
    
    parser.add_argument(
        '--query-image',
        type=str,
        required=True,
        help='Path to query image',
    )
    
    parser.add_argument(
        '--query-mask',
        type=str,
        required=True,
        help='Path to query mask',
    )
    
    parser.add_argument(
        '--search-image',
        type=str,
        default=None,
        help='Path to search image (for VQ2D)',
    )
    
    parser.add_argument(
        '--search-video',
        type=str,
        default=None,
        help='Path to search video for VQ3D',
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./inference_output',
        help='Directory to save outputs',
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations',
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save predictions as for VQ3D',
    )
    
    return parser.parse_args()


def load_image(image_path):
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def load_mask(mask_path):
    """Load and preprocess mask."""
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)
    mask = (mask > 128).astype(np.float32)
    return mask


def load_video(video_path, max_frames=None):
    """Load video frames."""
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    
    return np.stack(frames)


def preprocess_inputs(query_image, query_mask, search_data, transform):
    """Preprocess inputs for model."""
    # Apply transforms
    query_image = transform(query_image)
    query_mask = torch.from_numpy(query_mask).unsqueeze(0).float()
    
    if search_data.ndim == 3:
        # Single image
        search_images = transform(search_data).unsqueeze(0)
    else:
        # Video
        search_images = []
        for frame in search_data:
            search_images.append(transform(frame))
        search_images = torch.stack(search_images)
    
    # Add batch dimension
    query_image = query_image.unsqueeze(0)
    query_mask = query_mask.unsqueeze(0)
    
    if search_images.dim() == 4:
        search_images = search_images.unsqueeze(0)
    
    return query_image, query_mask, search_images


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Check inputs
    if args.search_image is None and args.search_video is None:
        raise ValueError("Must specify either --search-image or --search-video")
    
    # Setup configuration
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build model
    print("Building model...")
    model = build_model(cfg)
    model = model.to(device)
    
    # Load weights
    print(f"Loading weights from {args.weights}")
    load_checkpoint(
        checkpoint_path=args.weights,
        model=model,
        device=device,
    )
    
    model.eval()
    
    # Build transforms
    transform = build_transforms(cfg, is_train=False)
    
    # Load inputs
    print("Loading inputs...")
    query_image = load_image(args.query_image)
    query_mask = load_mask(args.query_mask)
    
    if args.search_image:
        search_data = load_image(args.search_image)
        is_video = False
    else:
        search_data = load_video(args.search_video, max_frames=cfg.DATALOADER.NUM_FRAMES)
        is_video = True
    
    print(f"Query image shape: {query_image.shape}")
    print(f"Search data shape: {search_data.shape}")
    
    # Preprocess
    query_image_t, query_mask_t, search_images_t = preprocess_inputs(
        query_image, query_mask, search_data, transform
    )
    
    # Move to device
    query_image_t = query_image_t.to(device)
    query_mask_t = query_mask_t.to(device)
    search_images_t = search_images_t.to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(
            query_images=query_image_t,
            query_masks=query_mask_t,
            search_images=search_images_t,
        )
    
    predictions = outputs['predictions']
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Save predictions
    pred_save_path = os.path.join(args.output_dir, 'predictions.npy')
    np.save(pred_save_path, predictions.cpu().numpy())
    print(f"Predictions saved to {pred_save_path}")
    
    # Visualize
    if args.visualize:
        print("Generating visualizations...")
        
        if is_video:
            # Visualize first frame
            vis_path = os.path.join(args.output_dir, 'visualization_frame0.png')
            visualize_predictions(
                query_image=query_image_t[0],
                query_mask=query_mask_t[0],
                search_image=search_images_t[0],
                prediction=predictions[0],
                save_path=vis_path,
            )
            print(f"Visualization saved to {vis_path}")
            
            # Save video if requested
            if args.save_video:
                video_path = os.path.join(args.output_dir, 'predictions.mp4')
                save_video_predictions(
                    search_images=search_images_t[0],
                    predictions=predictions[0],
                    save_path=video_path,
                )
                print(f"Video saved to {video_path}")
        else:
            vis_path = os.path.join(args.output_dir, 'visualization.png')
            visualize_predictions(
                query_image=query_image_t[0],
                query_mask=query_mask_t[0],
                search_image=search_images_t[0, 0],
                prediction=predictions[0, 0],
                save_path=vis_path,
            )
            print(f"Visualization saved to {vis_path}")
    
    print("Inference completed!")


if __name__ == '__main__':
    main()