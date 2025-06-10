# Semantic Segmentation in Aviary with SegFormer

This project implements semantic segmentation for an aviary using SegFormer, identifying 8 classes: Background, Branch, Camera, Fence, Ground, Nest, Tree, and Water.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Training from Scratch](#training-from-scratch)
  - [Using Pretrained Model](#using-pretrained-model)
  - [Inference Only](#inference-only)
- [Model Classes](#model-classes)
- [Evaluation](#evaluation)
- [Export Results](#export-results)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Roboflow account (for dataset download)

### Install Dependencies

```bash
pip install pytorch-lightning transformers datasets torch torchvision
pip install roboflow opencv-python pillow numpy pandas matplotlib tqdm
pip install tensorboard evaluate
```

## Project Structure

```
bird-segmentation/
│
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── config.py                    # Configuration settings
├── dataset.py                   # Dataset class
├── model.py                     # SegFormer model wrapper
├── train.py                     # Training script
├── inference.py                 # Inference script
├── evaluate.py                  # Evaluation script
├── export_results.py            # Export predictions to JSON
├── interpolation.py             # Mask interpolation utilities
├── utils.py                     # Utility functions
│
├── checkpoints/                 # Saved model checkpoints
│   └── bird_project_combined_v3.ckpt
│
└── lightning_logs/              # Training logs
```

## Dataset Setup

### Option 1: Download from Roboflow
```python
import roboflow
from roboflow import Roboflow

roboflow.login()
rf = Roboflow()

project = rf.workspace("bird-project-8tpx7").project("cameras_combined")
dataset = project.version(4).download("png-mask-semantic")
```

### Option 2: Use Custom Dataset
Organize your dataset as follows:
```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image1.png (mask)
│   └── _classes.csv
├── valid/
│   ├── image1.jpg
│   ├── image1.png (mask)
│   └── _classes.csv
└── test/
    ├── image1.jpg
    ├── image1.png (mask)
    └── _classes.csv
```

The `_classes.csv` should contain:
```csv
class_id,class_name
0,background
1,Branch
2,Camera
3,Fence
4,Ground
5,Nest
6,Tree
7,Water
```

## Usage

### Training from Scratch

1. **Configure training parameters** in `config.py`:
```python
# Edit config.py
BATCH_SIZE = 96
NUM_WORKERS = 12
MAX_EPOCHS = 100
LEARNING_RATE = 2e-5
IMAGE_SIZE = 128
```

2. **Run training**:
```bash
python train.py --dataset_path /path/to/dataset --gpus 1
```

Optional arguments:
- `--batch_size`: Batch size (default: 96)
- `--epochs`: Number of epochs (default: 100)
- `--checkpoint`: Resume from checkpoint
- `--patience`: Early stopping patience (default: 5)

### Using Pretrained Model

1. **Download pretrained checkpoint**:
Place `bird_project_combined_v3.ckpt` in the `checkpoints/` directory.

2. **Fine-tune on your data**:
```bash
python train.py --dataset_path /path/to/dataset --checkpoint checkpoints/bird_project_combined_v3.ckpt --epochs 50
```

3. **Evaluate the model**:
```bash
python evaluate.py --checkpoint checkpoints/bird_project_combined_v3.ckpt --dataset_path /path/to/dataset
```

### Inference Only

1. **Run inference on a single image**:
```bash
python inference.py --checkpoint checkpoints/bird_project_combined_v3.ckpt --image path/to/image.jpg --output output_overlay.png
```

2. **Run inference on a directory**:
```bash
python inference.py --checkpoint checkpoints/bird_project_combined_v3.ckpt --input_dir /path/to/images --output_dir /path/to/outputs
```

3. **With interpolation** (fills gaps in predictions):
```bash
python inference.py --checkpoint checkpoints/bird_project_combined_v3.ckpt --input_dir /path/to/images --output_dir /path/to/outputs --interpolate
```

## Model Classes

The model predicts 8 classes:

| ID | Class | Color (RGB) | Description |
|----|-------|-------------|-------------|
| 0 | Background | (0, 0, 0) | Black - Non-annotated regions |
| 1 | Branch | (255, 0, 0) | Red - Tree branches |
| 2 | Camera | (0, 255, 0) | Green - Camera equipment |
| 3 | Fence | (0, 0, 255) | Blue - Fencing structures |
| 4 | Ground | (255, 255, 0) | Yellow - Ground surface |
| 5 | Nest | (128, 0, 128) | Purple - Bird nests |
| 6 | Tree | (0, 255, 255) | Cyan - Tree trunks/foliage |
| 7 | Water | (255, 165, 0) | Orange - Water bodies |

## Evaluation

Run comprehensive evaluation:
```bash
python evaluate.py --checkpoint checkpoints/bird_project_combined_v3.ckpt --dataset_path /path/to/dataset --save_metrics
```

This will output:
- Overall mIoU and accuracy
- Per-class IoU scores
- Confusion matrix
- Sample predictions

### With Interpolation Comparison
```bash
python evaluate.py --checkpoint checkpoints/bird_project_combined_v3.ckpt --dataset_path /path/to/dataset --compare_interpolation
```

## Export Results

Export predictions in COCO format:
```bash
python export_results.py --checkpoint checkpoints/bird_project_combined_v3.ckpt --input_dir /path/to/images --output_dir exports/
```

This creates:
- `exports/overlays/`: Overlay visualizations
- `exports/masks/`: Segmentation masks
- `exports/results.json`: COCO-format annotations

## Advanced Features

### Mask Interpolation

The interpolation module fills gaps and smooths predictions using morphological operations:

```python
from interpolation import interpolate_mask

# Standard interpolation
interpolated = interpolate_mask(predicted_mask, kernel_size=15, threshold=5)

# Aggressive interpolation for filling large gaps
interpolated = interpolate_mask_dramatic(predicted_mask)
```

### Custom Training Configuration

For advanced users, modify `model.py`:
- Change backbone: `nvidia/segformer-b0` to `b1`, `b2`, etc.
- Adjust loss functions
- Add custom metrics

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Reduce `IMAGE_SIZE` (default: 128)
- Use gradient accumulation

### Poor Performance
- Increase training epochs
- Adjust learning rate
- Use data augmentation
- Check class imbalance in dataset

### Inference Speed
- Use smaller backbone (b0 is fastest)
- Batch process images
- Reduce image size

## Citation

This is an implementation of Segformer. The original code can be found at:
```bibtex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```
