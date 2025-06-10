"""
Utility functions for the segmentation project
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from config import COLOR_MAP


def prediction_to_vis(prediction):
    """
    Convert a segmentation mask (with integer class labels) into a color image.
    
    Args:
        prediction: numpy array with integer class labels
        
    Returns:
        PIL Image with color visualization
    """
    vis_shape = prediction.shape + (3,)
    vis = np.zeros(vis_shape, dtype=np.uint8)
    
    for class_id, color in COLOR_MAP.items():
        vis[prediction == class_id] = color
        
    return Image.fromarray(vis)


def plot_segmentation_comparison(original_image, predicted_mask, ground_truth_mask=None):
    """
    Plot segmentation results for comparison
    
    Args:
        original_image: PIL Image or numpy array
        predicted_mask: numpy array with predicted labels
        ground_truth_mask: optional numpy array with ground truth labels
    """
    num_plots = 3 if ground_truth_mask is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Predicted mask
    pred_vis = prediction_to_vis(predicted_mask)
    axes[1].imshow(pred_vis)
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    # Ground truth mask if provided
    if ground_truth_mask is not None:
        gt_vis = prediction_to_vis(ground_truth_mask)
        axes[2].imshow(gt_vis)
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_overlay(original_image, mask, alpha=0.5):
    """
    Create an overlay of the segmentation mask on the original image
    
    Args:
        original_image: PIL Image
        mask: numpy array with class labels
        alpha: transparency factor for overlay
        
    Returns:
        PIL Image with overlay
    """
    # Convert mask to visualization
    mask_vis = prediction_to_vis(mask)
    
    # Ensure both images are the same size
    mask_vis = mask_vis.resize(original_image.size)
    
    # Convert to RGBA
    original_rgba = original_image.convert("RGBA")
    mask_rgba = mask_vis.convert("RGBA")
    
    # Create overlay
    overlay = Image.blend(original_rgba, mask_rgba, alpha)
    
    return overlay


def calculate_class_distribution(mask):
    """
    Calculate the distribution of classes in a segmentation mask
    
    Args:
        mask: numpy array with class labels
        
    Returns:
        dict: class_id -> percentage of pixels
    """
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    distribution = {}
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        distribution[int(class_id)] = percentage
        
    return distribution


def print_class_distribution(mask, id2label=None):
    """
    Print the class distribution in a formatted way
    
    Args:
        mask: numpy array with class labels
        id2label: optional dict mapping class IDs to names
    """
    from config import ID2LABEL
    
    if id2label is None:
        id2label = ID2LABEL
        
    distribution = calculate_class_distribution(mask)
    
    print("\nClass Distribution:")
    print("-" * 40)
    for class_id, percentage in sorted(distribution.items()):
        class_name = id2label.get(class_id, f"Unknown_{class_id}")
        print(f"{class_name:15s}: {percentage:6.2f}%")
    print("-" * 40)