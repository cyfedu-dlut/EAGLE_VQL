# EAGLE: Episodic Appearance- and Geometry-aware Memory for Unified 2D-3D Visual Query Localization in Egocentric Vision

Official PyTorch implementation of **EAGLE**.

## 📋 Overview

EAGLE is a novel framework for visual query localization in egocentric videos, featuring:

- **🎯 Dual-Branch Architecture**: AMM (appearance) + GLM (geometry)
- **🧠 Meta-Learning Memory**: Adaptive learning with pseudo-label modulation
- **📐 3D Localization**: Multi-view aggregation with VGGT integration
- **🚀 State-of-the-Art**: Top performance on Ego4D VQ2D & VQ3D benchmarks

## 🚧 TODO List
- [x] Release the codebase(core modules)
- [x] Add inference code
- [x] Code optimization
- [x] Add re-organized test code
- [x] Optimize the json file for readable structure 
- [x] Support for additional backbones (SAM, CLIP)
- [ ] Real-time inference optimization
- [ ] Mobile deployment support
- [ ] Integration with robotics frameworks
- [ ] Additional datasets support
- [ ] Model compression and quantization


## 🔧 Installation

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
Follow [here](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D) and organize the data as following structure

```bash
# Download Ego4D videos and annotations
bash scripts/download_ego4d.sh --output_dir data/ego4d

# Structure
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
#### 2. Process annotations
```bash
python tools/prepare_ego4d_data.py \
    --data-dir ./data/ego4d \
    --output-dir ./data/ego4d_processed
```

### 📥 Download Pretrained Weights of DINOv2
```bash
bash scripts/download_dinov2.sh
```

### 🎯 Training
#### VQ2D
```bash
# Single GPU
python tools/train.py \
    --config-file configs/vq2d_base.yaml \
    --task vq2d

# Multi-GPU (Distributed)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train.py \
    --config-file configs/vq2d_base.yaml \
    --task vq2d \
    --num-gpus 4
```
#### VQ3D
```bash
# Single GPU
python tools/train.py \
    --config-file configs/vq3d_base.yaml \
    --task vq3d

# Multi-GPU
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train.py \
    --config-file configs/vq3d_base.yaml \
    --task vq3d \
    --num-gpus 4
```

### Custom Configuration
You can override config options via command line:
```bash
python tools/train.py \
    --config-file configs/vq2d_base.yaml \
    --task vq2d \
    SOLVER.BASE_LR 0.0002 \
    SOLVER.MAX_EPOCHS 100 \
    DATALOADER.BATCH_SIZE 16
```

### 🧪 Quick Evaluation
#### Evaluate VQ2D
```bash
python tools/test.py \
    --config-file configs/vq2d_base.yaml \
    --weights output/vq2d/checkpoints/best_model.pth \
    --task vq2d \
    --save-predictions \
    --visualize
```
#### Evaluate VQ3D
```bash
python tools/test.py \
    --config-file configs/vq3d_base.yaml \
    --weights output/vq3d/checkpoints/best_model.pth \
    --task vq3d \
    --save-predictions \
    --visualize
```
### 🔮 Quick Inference
```bash
python tools/inference.py \
    --config-file configs/vq2d_base.yaml \
    --weights output/vq2d/checkpoints/best_model.pth \
    --query-image examples/query.jpg \
    --query-mask examples/query_mask.png \
    --search-image examples/search.jpg \
    --visualize
```

### 📊 Monitor Training
```bash
# View tensorboard logs
tensorboard --logdir output/vq2d/tensorboard
```

### 📁 Project Structure
```bash
eagle/
├── configs/                    # Configuration files
│   ├── vq2d_base.yaml
│   └── vq3d_base.yaml
├── eagle/                      # Main package
│   ├── config/                 # Configuration system
│   ├── data/                   # Data loading and processing
│   │   ├── datasets/           # Dataset implementations
│   │   └── transforms/         # Data augmentation
│   ├── modeling/               # Model architectures
│   │   ├── backbone/           # Backbone networks
│   │   ├── memory/             # Memory modules (AMM & GLM)
│   │   ├── decoder/            # Decoder networks
│   │   ├── temporal/           # Temporal modeling
│   │   └── meta_arch/          # Meta architectures
│   ├── engine/                 # Training and evaluation
│   └── utils/                  # Utility functions
├── tools/                      # Scripts
│   ├── train.py               # Training script
│   ├── test.py                # Evaluation script
│   └── inference.py           # Inference script
└── tests/                      # Unit tests
```

### 📝 License
This project is licensed under the MIT License

### 🙏Acknowledgements
The codebase relies on some great repositories: [Ego4D-VQ2D](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D), [Ego4D-VQ3D](https://github.com/EGO4D/episodic-memory/blob/main/VQ3D), [DINOv2](https://github.com/facebookresearch/dinov2), [Segment Anything\(SAM\)](https://github.com/facebookresearch/segment-anything), [VGGT](https://github.com/facebookresearch/vggt), [PyTracking](https://github.com/visionml/pytracking) and many other inspiring works in the community. -->

### 📧 Contact
For questions and feedback, please contact:
Email: yfcao@mail.dlut.edu.cn

<!-- ### 📝Citation
```bibtex
@inproceedings{eagle2025,
  title={},
  author={},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2025}
}
``` -->
