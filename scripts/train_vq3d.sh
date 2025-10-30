#!/bin/bash
# Training script for VQ3D

# Configuration
CONFIG_FILE="configs/vq3d_base.yaml"
NUM_GPUS=4
OUTPUT_DIR="./output/vq3d"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
if [ $NUM_GPUS -gt 1 ]; then
    echo "Running distributed training on $NUM_GPUS GPUs..."
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        tools/train.py \
        --config-file $CONFIG_FILE \
        --task vq3d \
        --num-gpus $NUM_GPUS \
        OUTPUT.DIR $OUTPUT_DIR
else
    echo "Running single GPU training..."
    python tools/train.py \
        --config-file $CONFIG_FILE \
        --task vq3d \
        OUTPUT.DIR $OUTPUT_DIR
fi