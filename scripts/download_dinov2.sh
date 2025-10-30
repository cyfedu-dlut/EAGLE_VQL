#!/bin/bash
# Download DINOv2 pretrained weights

echo "Downloading DINOv2 pretrained weights..."

# Create weights directory
mkdir -p weights/dinov2

# DINOv2 ViT-S/14
echo "Downloading DINOv2 ViT-S/14..."
wget -O weights/dinov2/dinov2_vits14_pretrain.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth

# DINOv2 ViT-B/14
echo "Downloading DINOv2 ViT-B/14..."
wget -O weights/dinov2/dinov2_vitb14_pretrain.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# DINOv2 ViT-L/14
echo "Downloading DINOv2 ViT-L/14..."
wget -O weights/dinov2/dinov2_vitl14_pretrain.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

# DINOv2 ViT-G/14
echo "Downloading DINOv2 ViT-G/14..."
wget -O weights/dinov2/dinov2_vitg14_pretrain.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth

echo "Download completed!"