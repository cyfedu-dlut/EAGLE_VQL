"""
Script to prepare Ego4D dataset for EAGLE training.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare Ego4D dataset')
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to Ego4D dataset directory',
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed data',
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='vq2d',
        choices=['vq2d', 'vq3d'],
        help='Task type',
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to process',
    )
    
    return parser.parse_args()


def load_annotations(annotation_file):
    """Load annotations from JSON file."""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def process_vq2d_annotations(annotations, data_dir, output_dir):
    """Process VQ2D annotations."""
    print("Processing VQ2D annotations...")
    
    processed = []
    
    for item in tqdm(annotations):
        video_uid = item['video_uid']
        query_frame = item['query_frame']
        response_track = item.get('response_track', [])
        
        # Create output structure
        processed_item = {
            'video_uid': video_uid,
            'query_frame': query_frame,
            'query_bbox': item.get('query_bbox'),
            'response_track': response_track,
            'video_path': os.path.join(data_dir, 'clips', f"{video_uid}.mp4"),
        }
        
        processed.append(processed_item)
    
    return processed


def process_vq3d_annotations(annotations, data_dir, output_dir):
    """Process VQ3D annotations."""
    print("Processing VQ3D annotations...")
    
    processed = []
    
    for item in tqdm(annotations):
        video_uid = item['video_uid']
        query_frame = item['query_frame']
        response_track = item.get('response_track', [])
        
        # Create output structure
        processed_item = {
            'video_uid': video_uid,
            'query_frame': query_frame,
            'query_bbox': item.get('query_bbox'),
            'response_track': response_track,
            'video_path': os.path.join(data_dir, 'clips', f"{video_uid}.mp4"),
            'camera_poses': item.get('camera_poses', []),
        }
        
        processed.append(processed_item)
    
    return processed


def save_processed_annotations(processed, output_file):
    """Save processed annotations to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(processed, f, indent=2)
    
    print(f"Saved processed annotations to {output_file}")


def main():
    """Main function."""
    args = parse_args()
    
    # Paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    annotation_file = data_dir / 'v2' / 'annotations' / f'vq_{args.split}.json'
    
    if not annotation_file.exists():
        print(f"Annotation file not found: {annotation_file}")
        return
    
    print(f"Loading annotations from {annotation_file}")
    annotations = load_annotations(annotation_file)
    
    # Process annotations
    if args.task == 'vq2d':
        processed = process_vq2d_annotations(annotations, data_dir, output_dir)
    else:
        processed = process_vq3d_annotations(annotations, data_dir, output_dir)
    
    # Save processed annotations
    output_file = output_dir / f'{args.task}_{args.split}.json'
    save_processed_annotations(processed, output_file)
    
    print(f"Processing completed! Total items: {len(processed)}")


if __name__ == '__main__':
    main()