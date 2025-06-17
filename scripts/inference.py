"""
Inference script for SegFormer model
"""

### fixing model import issue
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
### end of fixing model import issue 

import argparse
import os
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerFeatureExtractor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SegformerFinetuner
from utils import prediction_to_vis
from interpolation import interpolate_mask, interpolate_mask_dramatic
from config import *


def run_inference_single(model, feature_extractor, image_path, interpolate=False, save_path=None):
    """Run inference on a single image"""
    # Load and prepare image
    input_image = Image.open(image_path).convert("RGB")
    width, height = input_image.size
    
    # Prepare image for model
    inputs = feature_extractor(images=input_image, return_tensors="pt")
    
    # Move model and inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
    
    # Convert to visualization
    mask_vis = prediction_to_vis(predicted_mask)
    mask_vis = mask_vis.resize((width, height)).convert("RGBA")
    
    # Create overlay
    input_image_rgba = input_image.convert("RGBA")
    overlay = Image.blend(input_image_rgba, mask_vis, alpha=0.5)
    
    # Save or display
    if save_path:
        overlay.save(save_path)
        print(f"Saved overlay to: {save_path}")
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.axis('off')
        plt.title(os.path.basename(image_path))
        plt.show()
    
    return predicted_mask, overlay


def run_inference_batch(model, feature_extractor, input_dir, output_dir, 
                       interpolate=False, dramatic_interpolation=False):
    """Run inference on a directory of images"""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    overlays_dir = os.path.join(output_dir, "overlays")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    
    for image_file in tqdm(image_files, desc="Running inference"):
        image_path = os.path.join(input_dir, image_file)
        input_image = Image.open(image_path).convert("RGB")
        width, height = input_image.size
        
        # Prepare image
        inputs = feature_extractor(images=input_image, return_tensors="pt")

        # Move input to GPU/CPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        if dramatic_interpolation:
            predicted_mask = interpolate_mask_dramatic(predicted_mask)
        elif interpolate:
            predicted_mask = interpolate_mask(predicted_mask)
        
        # Convert to visualization
        mask_vis = prediction_to_vis(predicted_mask)
        mask_vis = mask_vis.resize((width, height)).convert("RGBA")
        
        # Create overlay
        input_image_rgba = input_image.convert("RGBA")
        overlay = Image.blend(input_image_rgba, mask_vis, alpha=0.5)
        
        # Save outputs
        base_name = os.path.splitext(image_file)[0]
        overlay_path = os.path.join(overlays_dir, f"overlay_{base_name}.png")
        mask_path = os.path.join(masks_dir, f"mask_{base_name}.png")
        
        overlay.save(overlay_path)
        mask_vis.save(mask_path)
    
    print(f"Results saved to: {output_dir}")


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
    feature_extractor.size = args.image_size
    
    # Run inference
    if args.image:
        # Single image inference
        output_path = args.output if args.output else None
        run_inference_single(
            model, 
            feature_extractor, 
            args.image, 
            interpolate=args.interpolate,
            save_path=output_path
        )
    elif args.input_dir:
        # Batch inference
        if not args.output_dir:
            args.output_dir = os.path.join(args.input_dir, "segmentation_results")
        
        run_inference_batch(
            model,
            feature_extractor,
            args.input_dir,
            args.output_dir,
            interpolate=args.interpolate,
            dramatic_interpolation=args.dramatic_interpolation
        )
    else:
        print("Please provide either --image or --input_dir")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with SegFormer model")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                        help="Size to resize images")
    
    # Input arguments (mutually exclusive)
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image for inference")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing images for batch inference")
    
    # Output arguments
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for single image inference")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch inference")
    
    # Interpolation arguments
    parser.add_argument("--interpolate", action="store_true",
                        help="Apply interpolation to fill gaps in predictions")
    parser.add_argument("--dramatic_interpolation", action="store_true",
                        help="Apply dramatic interpolation for more visible effects")
    
    args = parser.parse_args()
    main(args)