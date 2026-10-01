# =====================================
# IMPROVED CONFIGURATION FILE
# =====================================

import os
import random
import numpy as np
import torch


class CFG:

    # =================================
    # PROJECT
    # =================================

    PROJECT_NAME = (
        "BreastCancerTransformer"
    )

    EXPERIMENT_NAME = (
        "ConvNeXt_ViT_Swin_IDC"
    )

    # =================================
    # ROOT PATHS
    # =================================

    ROOT_DIR = "project_data"

    RAW_DATASET_DIR = os.path.join(
        ROOT_DIR,
        "raw_dataset"
    )

    FINAL_DATASET_DIR = os.path.join(
        ROOT_DIR,
        "final_dataset"
    )

    # =================================
    # SPLITS
    # =================================

    TRAIN_DIR = os.path.join(
        FINAL_DATASET_DIR,
        "train"
    )

    VAL_DIR = os.path.join(
        FINAL_DATASET_DIR,
        "val"
    )

    TEST_DIR = os.path.join(
        FINAL_DATASET_DIR,
        "test"
    )

    # =================================
    # SAVE PATHS
    # =================================

    CHECKPOINT_DIR = "checkpoints"

    LOG_DIR = "logs"

    FIGURE_DIR = "figures"

    RESULT_DIR = "results"

    # =================================
    # CREATE DIRECTORIES
    # =================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        LOG_DIR,
        exist_ok=True
    )

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    # =================================
    # CLASSES
    # =================================

    CLASS_NAMES = [

        "IDC_negative",

        "IDC_positive"
    ]

    NUM_CLASSES = 1

    # =================================
    # IMAGE SETTINGS
    # =================================

    IMAGE_SIZE = 224

    PATCH_SIZE = 56

    NUM_PATCHES = 16

    CHANNELS = 3

    # =================================
    # DATASET SIZE
    # =================================

    TRAIN_PER_CLASS = 12000

    VAL_PER_CLASS = 2500

    TEST_PER_CLASS = 2500

    # =================================
    # DATALOADER
    # =================================

    BATCH_SIZE = 32

    NUM_WORKERS = 4

    PERSISTENT_WORKERS =  False

    PIN_MEMORY = True

    DROP_LAST = False

    SHUFFLE_TRAIN = True

    SHUFFLE_VAL = False

    SHUFFLE_TEST = False

    # =================================
    # TRAINING
    # =================================

    EPOCHS = 35

    LEARNING_RATE = 1e-4 

    MIN_LEARNING_RATE = 1e-6

    WEIGHT_DECAY =  1e-4

    EARLY_STOPPING_PATIENCE = 7

    SAVE_BEST_ONLY = True
    
    SAVE_EVERY_EPOCH = True

    GRADIENT_CLIP = 1.0

    # =================================
    # SCHEDULER
    # =================================

    USE_SCHEDULER = True

    SCHEDULER_FACTOR = 0.5

    SCHEDULER_PATIENCE = 1

    # =================================
    # MIXED PRECISION
    # =================================

    USE_AMP = True

    # =================================
    # DEVICE
    # =================================

    DEVICE = (

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    # =================================
    # RANDOMNESS
    # =================================

    SEED = 42

    # =================================
    # FEATURE DIMENSIONS
    # =================================

    FEATURE_DIM = 512

    # =================================
    # MODEL NAMES
    # =================================

    VIT_MODEL_NAME = (
        "vit_base_patch16_224"
    )

    SWIN_MODEL_NAME = (
        "swin_tiny_patch4_window7_224"
    )

    CONVNEXT_MODEL_NAME = (
        "convnext_tiny"
    )

    # =================================
    # IMAGE NORMALIZATION
    # =================================

    IMAGE_MEAN = [

        0.485,
        0.456,
        0.406
    ]

    IMAGE_STD = [

        0.229,
        0.224,
        0.225
    ]

    # =================================
    # TRAINING MODES
    # =================================

    CURRENT_MODEL = "fusion"

    AVAILABLE_MODELS = {

        "vit": "ViTModel",

        "swin": "SwinModel",

        "convnext": "ConvNeXtModel",

        "fusion": "FeatureFusionModel"
    }


    # =================================
    # GRADCAM SETTINGS
    # =================================

    GRADCAM_ALPHA = 0.35

    GRADCAM_COLORMAP = "turbo"

    # =================================
    # METRICS
    # =================================

    METRIC_NAMES = [

        "accuracy",

        "precision",

        "recall",

        "f1_score",

        "roc_auc"
    ]

    # =================================
    # CHECKPOINT NAMES
    # =================================

    BEST_MODEL_PATH = os.path.join(

        CHECKPOINT_DIR,

        "best_model.pth"
    )

    LAST_MODEL_PATH = os.path.join(

        CHECKPOINT_DIR,

        "last_model.pth"
    )

    # =================================
    # LOG FILES
    # =================================

    TRAIN_LOG_PATH = os.path.join(

        LOG_DIR,

        "train_log.csv"
    )

    METRICS_PATH = os.path.join(

        RESULT_DIR,

        "metrics.txt"
    )

    FINAL_RESULTS_PATH = os.path.join(

    RESULT_DIR,

    "final_results.txt"
    )

    # =================================
    # MODEL LIST
    # =================================

    MODEL_NAMES = [

        "vit",

        "swin",

        "convnext",

        "fusion"
    ]

    # =================================
    # FREEZE TRAINING
    # =================================

    FREEZE_EPOCHS = 2

    # =================================
    # MULTI MODEL RESULTS
    # =================================

    COMPARISON_RESULTS_PATH = os.path.join(

        RESULT_DIR,

        "model_comparison.csv"
    )

    # =================================
    # GRADCAM SETTINGS
    # =================================

    GRADCAM_SAMPLES_PER_CLASS = 4

    # =================================
    # VISUALIZATION
    # =================================

    SAVE_FIGURES_DPI = 300

    FIGURE_FORMAT = "png"

    # =================================
    # PERFORMANCE
    # =================================

    USE_WEIGHTED_SAMPLER = False

    LABEL_SMOOTHING = 0.02



# =====================================
# FIX RANDOM SEEDS
# =====================================

def set_seed(seed=CFG.SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    torch.backends.cudnn.allow_tf32 = True


# =====================================
# INITIALIZE SEED
# =====================================

set_seed()


