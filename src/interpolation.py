"""
Mask interpolation utilities for semantic segmentation
"""

import numpy as np
import cv2


def interpolate_mask(
    mask: np.ndarray,
    kernel_size: int = 40,
    threshold: int = 3,
    priority_order=None,
    kernel_shape: str = "ellipse",
    operations: str = "close",
    adaptive_kernel: bool = True,
    confidence_weighting: bool = True,
    boundary_aware: bool = True,
    preserve_original: bool = True
):
    """
    Advanced semantic segmentation mask interpolation and refinement.
    
    Args:
        mask: Input segmentation mask with integer class labels
        kernel_size: Size of structuring element (higher connects regions further apart)
        threshold: Confidence threshold (lower = more aggressive filling)
        priority_order: List of classes in order of increasing priority
                        If None, uses [0,4,6,2,7,3,5,1] for bird camera trap data
        kernel_shape: Shape of the structuring element ("ellipse", "rect", "cross")
        operations: Morphological operation ("close", "open", "dilate", "erode", "open_close", "close_open")
        adaptive_kernel: Whether to adjust kernel size based on region size
        confidence_weighting: Whether to use distance-based confidence
        boundary_aware: Whether to preserve original boundaries
        preserve_original: Whether to preserve original non-zero pixels
        
    Returns:
        np.ndarray: The interpolated segmentation mask
    """
    # Set default priority order for bird camera trap data if not specified
    if priority_order is None:
        # Background first (lowest priority), branch and camera at high priority
        priority_order = [0, 6, 4, 7, 2, 5, 1, 3]
    
    # Make a copy of the input mask
    original_mask = mask.copy()
    result_mask = np.zeros_like(mask, dtype=mask.dtype)
    
    # Create structuring element
    kernel_shapes = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS
    }
    
    if kernel_shape not in kernel_shapes:
        kernel_shape = "ellipse"  # Default to ellipse if invalid shape
    
    # Define operations
    ops = {
        "close": lambda img, k: cv2.morphologyEx(img, cv2.MORPH_CLOSE, k),
        "open": lambda img, k: cv2.morphologyEx(img, cv2.MORPH_OPEN, k),
        "dilate": lambda img, k: cv2.dilate(img, k),
        "erode": lambda img, k: cv2.erode(img, k),
        "open_close": lambda img, k: cv2.morphologyEx(
            cv2.morphologyEx(img, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k
        ),
        "close_open": lambda img, k: cv2.morphologyEx(
            cv2.morphologyEx(img, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k
        )
    }
    
    if operations not in ops:
        operations = "close"  # Default to close if invalid operation
    
    # Determine processing order
    unique_classes = np.unique(mask)
    if priority_order:
        # Ensure all classes in the mask are processed
        missing = sorted([cls for cls in unique_classes if cls not in priority_order])
        classes = missing + list(priority_order)
    else:
        classes = sorted(unique_classes)
    
    # Initialize confidence map for region competition
    confidence_map = np.zeros_like(mask, dtype=np.float32)
    
    # Extract boundaries if needed
    if boundary_aware:
        edges = np.zeros_like(mask, dtype=np.uint8)
        for cls in classes:
            if cls == 0:  # Skip background
                continue
            binary = (original_mask == cls).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edges, contours, -1, 1, 1)
    
    # Process each class
    for cls in classes:
        # Skip background if specified at index 0 and we have other classes
        if cls == 0 and classes[0] == 0 and len(classes) > 1:
            continue
            
        # Create binary mask for current class
        binary_mask = np.uint8(original_mask == cls) * 255
        
        # For adaptive kernel sizing
        if adaptive_kernel:
            # Calculate appropriate kernel size based on region size
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get average contour area
                avg_area = np.mean([cv2.contourArea(cnt) for cnt in contours])
                # Scale factor based on log of area
                area_factor = np.clip(np.log1p(avg_area) / 10, 0.5, 2.0)
                # Adjust kernel size (minimum 3)
                k_size = max(3, int(kernel_size * area_factor))
            else:
                k_size = kernel_size
        else:
            k_size = kernel_size
            
        # Create kernel for this class
        kernel = cv2.getStructuringElement(kernel_shapes[kernel_shape], (k_size, k_size))
        
        # Apply morphological operation
        processed = ops[operations](binary_mask, kernel)
        
        # Distance-based confidence if enabled
        if confidence_weighting:
            # Calculate distance transform (distance to nearest zero pixel)
            dist_transform = cv2.distanceTransform(processed, cv2.DIST_L2, 5)
            
            # Normalize distances to 0-1 range
            max_dist = np.max(dist_transform)
            if max_dist > 0:
                normalized_dist = dist_transform / max_dist
            else:
                normalized_dist = dist_transform
                
            # Weight by threshold
            class_confidence = normalized_dist * (processed > threshold)
        else:
            # Simple binary confidence
            class_confidence = np.float32(processed > threshold)
        
        # Update result based on confidence and priority
        update_mask = (class_confidence > confidence_map) & (processed > threshold)
        
        # Respect boundaries if boundary-aware
        if boundary_aware and cls > 0:  # Skip for background
            update_mask = update_mask & (edges == 0)
        
        result_mask[update_mask] = cls
        confidence_map[update_mask] = class_confidence[update_mask]
    
    # Optionally preserve original labels for non-zero regions
    if preserve_original:
        result_mask[original_mask > 0] = original_mask[original_mask > 0]
    
    return result_mask


def interpolate_mask_dramatic(mask):
    """
    Apply dramatic interpolation with very visible effects.
    Useful to verify that interpolation is actually being applied.
    
    Args:
        mask: Input segmentation mask
        
    Returns:
        Dramatically interpolated mask
    """
    return interpolate_mask(
        mask,
        kernel_size=18,
        threshold=3,
        operations="dilate",  # Very aggressive filling
        preserve_original=False,  # Don't preserve original pixels
        boundary_aware=False,     # Allow changes at boundaries
        adaptive_kernel=False     # Use fixed kernel size
    )


def visualize_interpolation_effect(original_mask, interpolated_mask):
    """
    Create a visualization showing the effect of interpolation
    
    Args:
        original_mask: Original segmentation mask
        interpolated_mask: Interpolated segmentation mask
        
    Returns:
        diff_mask: Binary mask showing changed pixels
        stats: Dictionary with statistics about the changes
    """
    # Calculate difference
    diff_mask = (original_mask != interpolated_mask).astype(np.uint8)
    
    # Calculate statistics
    total_pixels = original_mask.size
    changed_pixels = np.sum(diff_mask)
    percentage_changed = (changed_pixels / total_pixels) * 100
    
    # Per-class changes
    class_changes = {}
    for class_id in np.unique(original_mask):
        original_count = np.sum(original_mask == class_id)
        interpolated_count = np.sum(interpolated_mask == class_id)
        change = interpolated_count - original_count
        class_changes[int(class_id)] = {
            'original': int(original_count),
            'interpolated': int(interpolated_count),
            'change': int(change),
            'change_percentage': float(change / original_count * 100) if original_count > 0 else 0
        }
    
    stats = {
        'total_pixels': total_pixels,
        'changed_pixels': changed_pixels,
        'percentage_changed': percentage_changed,
        'class_changes': class_changes
    }
    
    return diff_mask, stats