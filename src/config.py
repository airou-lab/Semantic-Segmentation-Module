"""
Configuration settings for Bird Camera Trap Segmentation
"""

import os

# Model Configuration
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
NUM_CLASSES = 8
IMAGE_SIZE = 128

# Class definitions
ID2LABEL = {
    0: "background",
    1: "Branch",
    2: "Camera",
    3: "Fence",
    4: "Ground",
    5: "Nest",
    6: "Tree",
    7: "Water",
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Color map for visualization (RGB)
COLOR_MAP = {
    0: (0, 0, 0),       # Background - Black
    1: (255, 0, 0),     # Branch - Red
    2: (0, 255, 0),     # Camera - Green
    3: (0, 0, 255),     # Fence - Blue
    4: (255, 255, 0),   # Ground - Yellow
    5: (128, 0, 128),   # Nest - Purple
    6: (0, 255, 255),   # Tree - Cyan
    7: (255, 165, 0),   # Water - Orange
}

# Training Configuration
BATCH_SIZE = 96
NUM_WORKERS = 12
MAX_EPOCHS = 100
LEARNING_RATE = 2e-05
PATIENCE = 5
METRICS_INTERVAL = 10

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "lightning_logs")
EXPORT_DIR = os.path.join(PROJECT_ROOT, "exports")

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Roboflow Configuration (if using)
ROBOFLOW_WORKSPACE = "bird-project-8tpx7"
ROBOFLOW_PROJECT = "cameras_combined"
ROBOFLOW_VERSION = 4