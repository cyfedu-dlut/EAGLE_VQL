# EAGLE: Episodic Appearance- and Geometry-aware Memory for Unified 2D-3D Visual Query Localization in Egocentric Vision

Official PyTorch implementation of **EAGLE**.

## 📋 Overview

EAGLE is a novel framework for visual query localization in egocentric videos, featuring:

- **🎯 Dual-Branch Architecture**: AMM (appearance) + GLM (geometry)
- **🧠 Meta-Learning Memory**: Adaptive learning with pseudo-label modulation
- **📐 3D Localization**: Multi-view aggregation with VGGT integration
- **🚀 State-of-the-Art**: Top performance on Ego4D VQ2D & VQ3D benchmarks

## TODO List
- [ ] Init Code
- [ ] Add 2D inference code
- [ ] Code optimization
- [ ] Add re-organized test code
- [ ] Optimize the json file for readable structure 
- [ ] Set up Github pages 

<!-- ## 🔧 Installation

### Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.8

### Setup

```bash  
# Clone repository  
git clone https://github.com/cyfedu-dlut/EAGLE_VQL 
cd EAGLE 

# Create conda environment  
conda create -n eagle python=3.8  
conda activate eagle  

# Install dependencies  
pip install -r requirements.txt  

# Install EAGLE package  
pip install -e . 
```

### 📦Data Preparation
#### 1. Download the Ego4D Dataset
Follow [here](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D).

```bash
# Download Ego4D videos and annotations
bash scripts/download_ego4d.sh --output_dir data/ego4d

# structure:
# data/ego4d/
# ├── v2/
# │   ├── annotations/
# │   │   ├── vq_train.json
# │   │   ├── vq_val.json
# │   │   └── vq_test_unannotated.json
# │   └── videos/
# │       ├── {video_uid}.mp4
# │       └── ...
```

#### 2. Extract Video Clips
```bash
python scripts/extract_clips.py \
    --input_dir data/ego4d/v2/videos \
    --annot_path data/ego4d/v2/annotations/vq_train.json \
    --output_dir data/ego4d/v2/clips \
    --num_workers 16
```

#### 3.Preprocess Annotations
```bash
python scripts/preprocess_annotations.py \
    --input_path data/ego4d/v2/annotations/vq_train.json \
    --output_path data/ego4d/v2/annotations/vq_train_processed.json \
    --clip_dir data/ego4d/v2/clips
```

### 🎓Training: Train VQ2D Model
```bash
# Single GPU
python tools/train_vq2d.py --config configs/vq2d_train.yaml

# Multi-GPU (DDP)
python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train_vq2d.py --config configs/vq2d_train.yaml
```
### 📊Evaluation
#### VQ2D Evaluation
```bash
python tools/eval_vq2d.py \
    --config configs/vq2d_eval.yaml \
    --checkpoint checkpoints/eagle_vq2d_best.pth \
    --split val
```
#### VQ3D Evaluation
```bash
python tools/eval_vq3d.py \
    --config configs/vq3d_eval.yaml \
    --checkpoint checkpoints/eagle_vq2d_best.pth \
    --split val \
    --vggt_model_path checkpoints/vggt.pth
```
### 🎯Inference 
#### Quick Demo
```python
from eagle import EAGLE_VQL2D
import cv2
import torch

# Load model
model = EAGLE_VQL2D(feature_dim=768, memory_size=50)
model.load_state_dict(torch.load('checkpoints/eagle_vq2d_best.pth'))
model.eval()

# Load query and video
query_image = cv2.imread('query.jpg')
query_mask = cv2.imread('query_mask.png', 0)
video_frames = [...]  # List of video frames

# Initialize with query
query_tensor = preprocess_image(query_image)
mask_tensor = preprocess_mask(query_mask)
model.initialize(query_tensor, mask_tensor)

# Process video
results = []
for frame in video_frames:
    frame_tensor = preprocess_image(frame)
    pred_mask, confidence = model(frame_tensor)
    results.append({'mask': pred_mask, 'confidence': confidence})
```

#### Command Line Inference
```bash
python tools/inference.py \
    --checkpoint checkpoints/eagle_vq2d_best.pth \
    --query_image examples/query.jpg \
    --query_bbox 100,100,200,200 \
    --video examples/video.mp4 \
    --output results/
```

### 📁 Project Structure
```bash
EAGLE/
├── configs/              # Configuration files
├── eagle/                # Core package
│   ├── models/          # Model implementations
│   ├── data/            # Dataset & data loading
│   ├── evaluation/      # Evaluation metrics
│   └── utils/           # Utilities
├── scripts/             # Data preparation scripts
├── tools/               # Training & evaluation tools
└── notebooks/           # Jupyter notebooks
```

### 🙏Acknowledgements
The codebase relies on some great repositories: [Ego4D-VQ2D](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D), [Ego4D-VQ3D](https://github.com/EGO4D/episodic-memory/blob/main/VQ3D), [DINOv2](https://github.com/facebookresearch/dinov2), [Segment Anything\(SAM\)](https://github.com/facebookresearch/segment-anything), [VGGT](https://github.com/facebookresearch/vggt) and many other inspiring works in the community. -->

<!-- ### 📝Citation
```bibtex
@inproceedings{eagle2025,
  title={},
  author={},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2025}
}
``` -->
