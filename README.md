# EAGLE: Episodic Appearance- and Geometry-aware Memory for Unified 2D-3D Visual Query Localization in Egocentric Vision

🎉Our work has been accepted to AAAI 2026 Oral!

![alt text](image.png)

## 📋 Overview

EAGLE is a novel framework for visual query localization in egocentric videos, featuring:

- **🎯 Dual-Branch Architecture**: AMM (appearance) + GLM (geometry)
- **🧠 Meta-Learning Memory**: Adaptive learning with pseudo-label modulation
- **📐 3D Localization**: Multi-view aggregation with VGGT integration
- **🚀 State-of-the-Art**: Top performance on Ego4D VQ2D & VQ3D benchmarks

## 🚧 TODO List
- [x] Release the codebase(core modules)
- [x] Code optimization
- [x] Optimize the json file for readable structure 
- [] Training and inference code
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


### 🙏Acknowledgements
The codebase relies on some great repositories: [Ego4D-VQ2D](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D), [Ego4D-VQ3D](https://github.com/EGO4D/episodic-memory/blob/main/VQ3D), [DINOv2](https://github.com/facebookresearch/dinov2), [Segment Anything\(SAM\)](https://github.com/facebookresearch/segment-anything), [VGGT](https://github.com/facebookresearch/vggt), [PyTracking](https://github.com/visionml/pytracking) and many other inspiring works in the community. 

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
