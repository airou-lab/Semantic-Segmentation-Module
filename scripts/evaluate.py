"""
Evaluation script for SegFormer model
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerFeatureExtractor
from datasets import load_metric
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from dataset import SemanticSegmentationDataset
from model import SegformerFinetuner
from interpolation import interpolate_mask, interpolate_mask_dramatic
from utils import prediction_to_vis, plot_segmentation_comparison
from config import *


def evaluate_segmentation(pred_masks, true_masks, num_classes):
    """Evaluate segmentation metrics for batches"""
    metric = load_metric("mean_iou")
    
    # Ensure masks are numpy arrays
    pred_masks = np.array(pred_masks)
    true_masks = np.array(true_masks)
    
    # Add batch
    metric.add_batch(predictions=pred_masks, references=true_masks)
    
    # Compute IoU metrics
    result = metric.compute(
        num_labels=num_classes,
        ignore_index=255,
        reduce_labels=False
    )
    
    mean_iou = result["mean_iou"]
    per_class_iou = result["per_category_iou"]
    accuracy = np.mean(pred_masks.flatten() == true_masks.flatten())
    
    return mean_iou, per_class_iou, accuracy


def evaluate_model(model, test_dataloader, device, compare_interpolation=False):
    """Evaluate model performance on test set"""
    model.eval()
    model = model.to(device)
    
    num_classes = model.num_classes
    class_labels = [model.id2label[i] for i in range(num_classes)]
    
    # Metrics containers
    results = defaultdict(lambda: {"mean_iou": [], "accuracy": [], "per_class_iou": []})
    
    # Evaluate with and without interpolation
    print("Running evaluation...")
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        images, true_masks = batch['pixel_values'], batch['labels']
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model.model(pixel_values=images)
            logits = outputs.logits
            upsampled_logits = F.interpolate(
                logits, 
                size=true_masks.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            predicted_masks = upsampled_logits.argmax(dim=1).cpu().numpy()
            true_masks_np = true_masks.cpu().numpy()
            
            # Evaluate BEFORE interpolation
            mean_iou, per_class_iou, accuracy = evaluate_segmentation(
                predicted_masks, true_masks_np, num_classes
            )
            results["Before"]["mean_iou"].append(mean_iou)
            results["Before"]["accuracy"].append(accuracy)
            results["Before"]["per_class_iou"].append(per_class_iou)
            
            if compare_interpolation:
                # Interpolate masks
                interpolated_masks = np.array([
                    interpolate_mask(mask) for mask in predicted_masks
                ])
                
                # Evaluate AFTER interpolation
                mean_iou, per_class_iou, accuracy = evaluate_segmentation(
                    interpolated_masks, true_masks_np, num_classes
                )
                results["After"]["mean_iou"].append(mean_iou)
                results["After"]["accuracy"].append(accuracy)
                results["After"]["per_class_iou"].append(per_class_iou)
    
    # Aggregate results
    summary_results = {
        "Mean IoU": np.mean(results["Before"]["mean_iou"]),
        "Accuracy": np.mean(results["Before"]["accuracy"]),
    }
    
    if compare_interpolation:
        summary_results.update({
            "Mean IoU (Interpolated)": np.mean(results["After"]["mean_iou"]),
            "Accuracy (Interpolated)": np.mean(results["After"]["accuracy"]),
        })
    
    # Per-class IoU
    per_class_iou_before = np.mean(results["Before"]["per_class_iou"], axis=0)
    
    metrics_df = pd.DataFrame({
        "Class": class_labels,
        "IoU": per_class_iou_before,
    })
    
    if compare_interpolation:
        per_class_iou_after = np.mean(results["After"]["per_class_iou"], axis=0)
        metrics_df["IoU (Interpolated)"] = per_class_iou_after
        metrics_df["Improvement"] = per_class_iou_after - per_class_iou_before
    
    return summary_results, metrics_df


def visualize_predictions(model, test_dataset, feature_extractor, device, num_samples=5):
    """Visualize sample predictions"""
    model.eval()
    model = model.to(device)
    
    # Randomly select samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Get image and mask
        batch = test_dataset[idx]
        image = batch['pixel_values'].unsqueeze(0).to(device)
        true_mask = batch['labels'].cpu().numpy()
        
        # Get prediction
        with torch.no_grad():
            outputs = model.model(pixel_values=image)
            logits = outputs.logits
            upsampled_logits = F.interpolate(
                logits,
                size=true_mask.shape,
                mode="bilinear",
                align_corners=False
            )
            predicted_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Load original image for visualization
        image_path = os.path.join(test_dataset.root_dir, test_dataset.images[idx])
        original_image = Image.open(image_path)
        
        # Plot
        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(prediction_to_vis(predicted_mask))
        axes[i, 1].set_title("Predicted Mask")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(prediction_to_vis(true_mask))
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig


def main(args):
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = SegformerFinetuner.load_from_checkpoint(
        args.checkpoint,
        map_location=torch.device("cpu")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)
    
    # Initialize feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = IMAGE_SIZE
    
    # Create test dataset
    test_dataset = SemanticSegmentationDataset(
        os.path.join(args.dataset_path, "test/"),
        feature_extractor
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Run evaluation
    summary_results, metrics_df = evaluate_model(
        model, 
        test_dataloader, 
        device,
        compare_interpolation=args.compare_interpolation
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print("\nOverall Metrics:")
    for metric, value in summary_results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPer-Class IoU:")
    print(metrics_df.to_string(index=False))
    
    # Save metrics if requested
    if args.save_metrics:
        metrics_path = os.path.join(EXPORT_DIR, "evaluation_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nMetrics saved to: {metrics_path}")
    
    # Visualize predictions if requested
    if args.visualize:
        print("\nGenerating sample predictions...")
        fig = visualize_predictions(
            model, 
            test_dataset, 
            feature_extractor, 
            device,
            num_samples=args.num_samples
        )
        
        if args.save_visualizations:
            viz_path = os.path.join(EXPORT_DIR, "sample_predictions.png")
            fig.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"Visualizations saved to: {viz_path}")
        else:
            plt.show()
    
    # Plot IoU comparison if interpolation was compared
    if args.compare_interpolation and args.visualize:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics_df))
        width = 0.35
        
        plt.bar(x - width/2, metrics_df['IoU'], width, label='Original')
        plt.bar(x + width/2, metrics_df['IoU (Interpolated)'], width, label='Interpolated')
        
        plt.xlabel('Class')
        plt.ylabel('IoU')
        plt.title('Per-Class IoU: Original vs Interpolated')
        plt.xticks(x, metrics_df['Class'], rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if args.save_visualizations:
            plt.savefig(os.path.join(EXPORT_DIR, "iou_comparison.png"), dpi=150)
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SegFormer model")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset")
    
    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    
    # Evaluation options
    parser.add_argument("--compare_interpolation", action="store_true",
                        help="Compare results with and without interpolation")
    parser.add_argument("--save_metrics", action="store_true",
                        help="Save evaluation metrics to CSV")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize sample predictions")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save visualizations instead of displaying")
    
    args = parser.parse_args()
    main(args)