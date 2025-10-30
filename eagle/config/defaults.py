"""
Default configuration for EAGLE.
This file defines all possible config options.
"""

from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "EAGLE_VQ2D"  # or "EAGLE_VQ3D"
_C.MODEL.WEIGHTS = ""  # Path to pretrained weights

# -----------------------------------------------------------------------------
# Backbone
# -----------------------------------------------------------------------------
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "dinov2_vitb14"
_C.MODEL.BACKBONE.PRETRAINED = True
_C.MODEL.BACKBONE.FREEZE = True
_C.MODEL.BACKBONE.FEATURE_DIM = 768
_C.MODEL.BACKBONE.PATCH_SIZE = 14
_C.MODEL.BACKBONE.USE_CHECKPOINT = False  # Gradient checkpointing

# Alternative backbones
_C.MODEL.BACKBONE.CLIP_MODEL = "ViT-B/16"

# -----------------------------------------------------------------------------
# AMM (Appearance-Aware Meta-Learning Memory)
# -----------------------------------------------------------------------------
_C.MODEL.AMM = CN()
_C.MODEL.AMM.ENABLED = True
_C.MODEL.AMM.MEMORY_SIZE = 50
_C.MODEL.AMM.PSEUDO_LABEL_CHANNELS = 32
_C.MODEL.AMM.META_LEARNER_ITERS = 3
_C.MODEL.AMM.KERNEL_SIZE = 3
_C.MODEL.AMM.REGULARIZER = 0.01
_C.MODEL.AMM.CONFIDENCE_THRESHOLD = 0.6
_C.MODEL.AMM.UPDATE_INTERVAL = 25

# -----------------------------------------------------------------------------
# GLM (Geometry-Aware Localization Memory)
# -----------------------------------------------------------------------------
_C.MODEL.GLM = CN()
_C.MODEL.GLM.ENABLED = True
_C.MODEL.GLM.MEMORY_SIZE = 50
_C.MODEL.GLM.DCF_ITERS = 5
_C.MODEL.GLM.LAMBDA_REG = 0.01
_C.MODEL.GLM.SCALE_FACTOR = 1.5
_C.MODEL.GLM.GAUSSIAN_SIGMA = 2.0
_C.MODEL.GLM.USE_STATIC = False  # Use static query memory
_C.MODEL.GLM.UPDATE_FIRST_N = 100  # Update memory for first N frames

# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.HIDDEN_DIM = 64
_C.MODEL.DECODER.OUTPUT_DIM = 1

# -----------------------------------------------------------------------------
# VGGT (for VQ3D)
# -----------------------------------------------------------------------------
_C.MODEL.VGGT = CN()
_C.MODEL.VGGT.ENABLED = False
_C.MODEL.VGGT.WEIGHTS = ""
_C.MODEL.VGGT.DEPTH_SCALE = 1.0
_C.MODEL.VGGT.UNCERTAINTY_THRESHOLD = 0.1
_C.MODEL.VGGT.MIN_VIEWS = 3

# Multi-view aggregation weights
_C.MODEL.VGGT.PHI = 0.333    # Weight for P_av
_C.MODEL.VGGT.PSI = 0.333    # Weight for P_lambda
_C.MODEL.VGGT.MU = 0.333     # Weight for P_max
_C.MODEL.VGGT.ZETA = 1.0     # Uncertainty weight
_C.MODEL.VGGT.LAMBDA_THRESHOLD = 0.6

# -----------------------------------------------------------------------------
# SAM (Segment Anything Model for query initialization)
# -----------------------------------------------------------------------------
_C.MODEL.SAM = CN()
_C.MODEL.SAM.ENABLED = False
_C.MODEL.SAM.MODEL_TYPE = "vit_h"
_C.MODEL.SAM.CHECKPOINT = "checkpoints/sam_vit_h.pth"
_C.MODEL.SAM.BBOX_SCALE = 0.67  # 2/3 bbox scale as in paper

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ("vq2d_train",)
_C.DATASETS.VAL = ("vq2d_val",)
_C.DATASETS.TEST = ("vq2d_test",)

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.ASPECT_RATIO_GROUPING = False
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
_C.DATALOADER.REPEAT_THRESHOLD = 0.0
_C.DATALOADER.PREFETCH_FACTOR = 2
_C.DATALOADER.PERSISTENT_WORKERS = True
_C.DATALOADER.PIN_MEMORY = True

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.IMAGE_SIZE = 448
_C.INPUT.MIN_SIZE_TRAIN = (448,)
_C.INPUT.MAX_SIZE_TRAIN = 448
_C.INPUT.MIN_SIZE_TEST = 448
_C.INPUT.MAX_SIZE_TEST = 448

# Clip sampling
_C.INPUT.CLIP_LENGTH = 16  # Number of frames per clip
_C.INPUT.STRIDE = 8
_C.INPUT.SAMPLING_STRATEGY = "uniform"  # or "random"

# Augmentation
_C.INPUT.AUGMENTATION = CN()
_C.INPUT.AUGMENTATION.ENABLED = True
_C.INPUT.AUGMENTATION.HORIZONTAL_FLIP = 0.5
_C.INPUT.AUGMENTATION.COLOR_JITTER = 0.4
_C.INPUT.AUGMENTATION.RANDOM_CROP = True

# Normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Solver (Optimizer and LR Scheduler)
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.BASE_LR = 0.0025
_C.SOLVER.WEIGHT_DECAY = 0.05
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.BETAS = (0.9, 0.999)

# Learning rate schedule
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
_C.SOLVER.WARMUP_ITERS = 2500
_C.SOLVER.WARMUP_FACTOR = 0.001
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.MAX_ITER = 25000

# Steps for MultiStepLR
_C.SOLVER.STEPS = (15000, 20000)
_C.SOLVER.GAMMA = 0.1

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN()
_C.SOLVER.CLIP_GRADIENTS.ENABLED = True
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"  # or "value"
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Gradient accumulation
_C.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1

# Mixed precision training
_C.SOLVER.AMP = CN()
_C.SOLVER.AMP.ENABLED = False

# Batch size
_C.SOLVER.IMS_PER_BATCH = 8  # Total batch size across all GPUs
_C.SOLVER.REFERENCE_WORLD_SIZE = 1  # For linear scaling

# Checkpoint period
_C.SOLVER.CHECKPOINT_PERIOD = 2500

# Max number of epochs (alternative to MAX_ITER)
_C.SOLVER.MAX_EPOCHS = 10

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.SEGMENTATION_WEIGHT = 1.0  # Weight for AMM segmentation loss
_C.LOSS.TRACKING_WEIGHT = 1.0      # Weight for GLM tracking loss (rho)
_C.LOSS.USE_LOVASZ = True          # Use Lovasz hinge loss
_C.LOSS.USE_HINGE = True           # Use hinge loss for DCF

# -----------------------------------------------------------------------------
# Test / Evaluation
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.EVAL_PERIOD = 2500  # Evaluation period during training
_C.TEST.IMS_PER_BATCH = 1   # Batch size for testing

# VQ2D evaluation
_C.TEST.VQ2D = CN()
_C.TEST.VQ2D.IOU_THRESHOLDS = [0.25, 0.5, 0.75]
_C.TEST.VQ2D.DISTANCE_THRESHOLD = 64  # pixels

# VQ3D evaluation
_C.TEST.VQ3D = CN()
_C.TEST.VQ3D.DISTANCE_THRESHOLDS = [0.2, 0.3, 0.5]  # meters
_C.TEST.VQ3D.USE_SIM3_ALIGNMENT = True
_C.TEST.VQ3D.MIN_VIEWS = 3

# Post-processing
_C.TEST.POSTPROCESS = CN()
_C.TEST.POSTPROCESS.APPLY_LAPLACIAN = True
_C.TEST.POSTPROCESS.LAPLACIAN_WINDOW = 100
_C.TEST.POSTPROCESS.LAPLACIAN_THRESHOLD = 100
_C.TEST.POSTPROCESS.MEDIAN_FILTER_WINDOW = 5
_C.TEST.POSTPROCESS.CONFIDENCE_RATIO = 0.8

# Output
_C.TEST.DETECTIONS_PER_IMAGE = 100

# -----------------------------------------------------------------------------
# Distributed training
# -----------------------------------------------------------------------------
_C.DISTRIBUTED = CN()
_C.DISTRIBUTED.BACKEND = "nccl"
_C.DISTRIBUTED.INIT_METHOD = "env://"

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = "./outputs"
_C.SEED = 42
_C.CUDNN_BENCHMARK = False
_C.VIS_PERIOD = 0  # Visualization period (0 to disable)

# Global config object
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0

# -----------------------------------------------------------------------------
# Ego4D specific
# -----------------------------------------------------------------------------
_C.EGO4D = CN()
_C.EGO4D.ROOT_DIR = "data/ego4d/v2"
_C.EGO4D.ANNOT_DIR = "${EGO4D.ROOT_DIR}/annotations"
_C.EGO4D.CLIP_DIR = "${EGO4D.ROOT_DIR}/clips"
_C.EGO4D.VERSION = "v2"

# Dataset splits
_C.EGO4D.VQ2D_TRAIN = "${EGO4D.ANNOT_DIR}/vq_train.json"
_C.EGO4D.VQ2D_VAL = "${EGO4D.ANNOT_DIR}/vq_val.json"
_C.EGO4D.VQ2D_TEST = "${EGO4D.ANNOT_DIR}/vq_test_unannotated.json"

_C.EGO4D.VQ3D_TRAIN = "${EGO4D.ANNOT_DIR}/vq_train.json"  # Same annotations
_C.EGO4D.VQ3D_VAL = "${EGO4D.ANNOT_DIR}/vq_val.json"
_C.EGO4D.VQ3D_TEST = "${EGO4D.ANNOT_DIR}/vq_test_unannotated.json"