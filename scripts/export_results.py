"""
Export segmentation results to various formats
"""

import argparse
import os
import json
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerFeatureExtractor
from tqdm import tqdm

from model import SegformerFinetuner
from utils import prediction_to_vis
from interpolation import interpolate_mask
from config import *


def export_to_coco_format(predictions_dict, output_path):
    """
    Export predictions to COCO format JSON
    
    Args:
        predictions_dict: Dictionary with image filenames as keys
        output_path: Path to save the JSON file
    """
    # Define categories (starting at 1, excluding background)
    categories = [
        {"id": i, "name": ID2LABEL[i]} 
        for i in range(1, NUM_CLASSES)
    ]
    
    coco_format = {}
    
    for image_file, data in predictions_dict.items():
        coco_format[image_file] = {
            "annotations": data["annotations"],
            "categories": categories,
            "image_info": data.get("image_info", {})
        }
    
    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=4)
    
    print(f"Exported COCO format annotations to: {output_path}")


def mask_to_polygons(mask, class_id, min_area=10):
    """
    Convert binary mask to polygon annotations
    
    Args:
        mask: Binary mask for a single class
        class_id: Class ID
        min_area: Minimum area threshold for polygons
        
    Returns:
        List of annotation dictionaries
    """
    annotations = []
    annotation_id = 0
    
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to polygon format
        segmentation = approx.flatten().tolist()
        if len(segmentation) < 6:  # Need at least 3 points
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        annotation = {
            "id": annotation_id,
            "category_id": int(class_id),
            "segmentation": [segmentation],
            "bbox": [x, y, w, h],
            "area": area,
            "iscrowd": 0
        }
        
        annotations.append(annotation)
        annotation_id += 1
    
    return annotations


def process_image(model, feature_extractor, image_path, interpolate=False):
    """
    Process a single image and return predictions
    
    Args:
        model: SegFormer model
        feature_extractor: Feature extractor
        image_path: Path to image
        interpolate: Whether to apply interpolation
        
    Returns:
        predicted_mask: Numpy array with class predictions
        overlay: PIL Image with overlay visualization
        annotations: List of polygon annotations
    """
    # Load image
    input_image = Image.open(image_path).convert("RGB")
    width, height = input_image.size
    
    # Prepare image
    inputs = feature_extractor(images=input_image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model.model(**inputs)
        logits = outputs.logits
    
    # Upsample to original size
    upsampled_logits = F.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )
    predicted_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    # Apply interpolation if requested
    if interpolate:
        predicted_mask = interpolate_mask(predicted_mask)
    
    # Create visualization
    mask_vis = prediction_to_vis(predicted_mask)
    mask_vis = mask_vis.resize((width, height)).convert("RGBA")
    
    # Create overlay
    input_image_rgba = input_image.convert("RGBA")
    overlay = Image.blend(input_image_rgba, mask_vis, alpha=0.5)
    
    # Extract polygon annotations
    annotations = []
    for class_id in range(1, NUM_CLASSES):  # Skip background
        binary_mask = (predicted_mask == class_id)
        class_annotations = mask_to_polygons(binary_mask, class_id)
        annotations.extend(class_annotations)
    
    # Update annotation IDs to be unique
    for i, ann in enumerate(annotations):
        ann["id"] = i + 1
    
    return predicted_mask, overlay, annotations


def main(args):
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = SegformerFinetuner.load_from_checkpoint(
        args.checkpoint,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.eval()
    
    # Initialize feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = IMAGE_SIZE
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    overlays_dir = os.path.join(args.output_dir, "overlays")
    masks_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get all image files
    image_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    print(f"Processing {len(image_files)} images...")
    
    # Process images and collect results
    export_dict = {}
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.input_dir, image_file)
        
        # Process image
        predicted_mask, overlay, annotations = process_image(
            model, 
            feature_extractor, 
            image_path,
            interpolate=args.interpolate
        )
        
        # Save outputs
        base_name = os.path.splitext(image_file)[0]
        
        # Save overlay
        overlay_filename = f"overlay_{base_name}.png"
        overlay_path = os.path.join(overlays_dir, overlay_filename)
        overlay.save(overlay_path)
        
        # Save mask
        mask_filename = f"mask_{base_name}.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        mask_vis = prediction_to_vis(predicted_mask)
        mask_vis.save(mask_path)
        
        # Save raw mask as numpy if requested
        if args.save_raw_masks:
            raw_mask_path = os.path.join(masks_dir, f"raw_{base_name}.npy")
            np.save(raw_mask_path, predicted_mask)
        
        # Add to export dictionary
        image = Image.open(image_path)
        export_dict[image_file] = {
            "annotations": annotations,
            "image_info": {
                "width": image.width,
                "height": image.height,
                "file_name": image_file
            }
        }
    
    # Export to JSON
    json_path = os.path.join(args.output_dir, "results.json")
    export_to_coco_format(export_dict, json_path)
    
    # Generate summary statistics if requested
    if args.generate_stats:
        stats = generate_statistics(export_dict)
        stats_path = os.path.join(args.output_dir, "statistics.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Statistics saved to: {stats_path}")
    
    print(f"\nExport completed. Results saved to: {args.output_dir}")


def generate_statistics(export_dict):
    """Generate statistics from the export dictionary"""
    stats = {
        "total_images": len(export_dict),
        "total_annotations": 0,
        "annotations_per_class": {ID2LABEL[i]: 0 for i in range(1, NUM_CLASSES)},
        "average_annotations_per_image": 0,
        "images_per_class": {ID2LABEL[i]: 0 for i in range(1, NUM_CLASSES)}
    }
    
    for image_file, data in export_dict.items():
        annotations = data["annotations"]
        stats["total_annotations"] += len(annotations)
        
        # Track which classes appear in this image
        classes_in_image = set()
        
        for ann in annotations:
            class_id = ann["category_id"]
            class_name = ID2LABEL[class_id]
            stats["annotations_per_class"][class_name] += 1
            classes_in_image.add(class_name)
        
        # Update images per class
        for class_name in classes_in_image:
            stats["images_per_class"][class_name] += 1
    
    # Calculate average
    if stats["total_images"] > 0:
        stats["average_annotations_per_image"] = (
            stats["total_annotations"] / stats["total_images"]
        )
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export segmentation results")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    
    # Processing options
    parser.add_argument("--interpolate", action="store_true",
                        help="Apply interpolation to predictions")
    parser.add_argument("--save_raw_masks", action="store_true",
                        help="Save raw prediction masks as numpy arrays")
    parser.add_argument("--generate_stats", action="store_true",
                        help="Generate statistics about the predictions")
    
    args = parser.parse_args()
    main(args)