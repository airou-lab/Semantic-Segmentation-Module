import os
import random
import numpy as np
import cv2
from tqdm import tqdm

# ---------------------------
# Configuration (Modify Paths)
# ---------------------------
DATASET_PATH = "/home/danielp_airlab/Documents/bird_project/scripts/cameras_combined-5/train"  # Full path to dataset
OUTPUT_PATH = "/home/danielp_airlab/Documents/bird_project/scripts/cameras_combined-5/augmented_train"  # Augmented data folder
BRANCH_CLASS_ID = 1  # Adjust if branch has a different ID
NUM_BRANCHES_PER_IMAGE = (2, 7)  # Randomly place between 2 to 5 branches

# Ensure output directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------------------
# Load Dataset
# ---------------------------
image_files = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith(".jpg")])
mask_files = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith("_mask.png")])

# Ensure each image has a corresponding mask
image_to_mask = {img: img.replace(".jpg", "_mask.png") for img in image_files if img.replace(".jpg", "_mask.png") in mask_files}

# Select 50% of training images randomly
num_samples = int(len(image_to_mask) * 0.5)
selected_images = random.sample(list(image_to_mask.keys()), num_samples)

# ---------------------------
# Process Selected Images
# ---------------------------
for img_filename in tqdm(selected_images, desc="Augmenting training data"):
    img_path = os.path.join(DATASET_PATH, img_filename)
    mask_path = os.path.join(DATASET_PATH, image_to_mask[img_filename])

    # Load image and mask
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Extract branch regions
    branch_mask = (mask == BRANCH_CLASS_ID).astype(np.uint8) * 255
    contours, _ = cv2.findContours(branch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store extracted branches
    branch_cutouts = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Ignore small branches
            continue

        # Create bounding box around the branch
        x, y, w, h = cv2.boundingRect(contour)
        branch_cutout = image[y:y+h, x:x+w]
        branch_mask_cutout = branch_mask[y:y+h, x:x+w]

        # Apply augmentation (rotate, flip, scale)
        if random.random() < 0.5:
            branch_cutout = cv2.flip(branch_cutout, 1)
            branch_mask_cutout = cv2.flip(branch_mask_cutout, 1)
        if random.random() < 0.5:
            branch_cutout = cv2.flip(branch_cutout, 0)
            branch_mask_cutout = cv2.flip(branch_mask_cutout, 0)

        angle = random.randint(-30, 30)  # Random rotation
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        branch_cutout = cv2.warpAffine(branch_cutout, M, (w, h))
        branch_mask_cutout = cv2.warpAffine(branch_mask_cutout, M, (w, h))

        branch_cutouts.append((branch_cutout, branch_mask_cutout))

    # Skip if no valid branches were found
    if not branch_cutouts:
        continue

    # ---------------------------
    # Paste Multiple Branches onto a New Random Image
    # ---------------------------
    target_img_filename = random.choice(list(image_to_mask.keys()))
    target_img_path = os.path.join(DATASET_PATH, target_img_filename)
    target_mask_path = os.path.join(DATASET_PATH, image_to_mask[target_img_filename])

    target_image = cv2.imread(target_img_path)
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)

    h_t, w_t, _ = target_image.shape
    num_branches_to_paste = random.randint(*NUM_BRANCHES_PER_IMAGE)

    for _ in range(num_branches_to_paste):
        branch_cutout, branch_mask_cutout = random.choice(branch_cutouts)
        h, w, _ = branch_cutout.shape

        # Select a random location in the target image
        rand_x = random.randint(0, w_t - w)
        rand_y = random.randint(0, h_t - h)

        # Overlay the branch cutout onto the target image
        roi = target_image[rand_y:rand_y+h, rand_x:rand_x+w]
        branch_mask_cutout_rgb = cv2.cvtColor(branch_mask_cutout, cv2.COLOR_GRAY2BGR)

        # Use alpha blending for realistic integration
        alpha = (branch_mask_cutout / 255.0).astype(np.float32)
        blended = (roi * (1 - alpha[..., None]) + branch_cutout * alpha[..., None]).astype(np.uint8)
        target_image[rand_y:rand_y+h, rand_x:rand_x+w] = blended

        # Update mask with branch class
        target_mask[rand_y:rand_y+h, rand_x:rand_x+w][branch_mask_cutout > 0] = BRANCH_CLASS_ID

    # ---------------------------
    # Save Augmented Image & Mask
    # ---------------------------
    aug_img_filename = f"aug_{target_img_filename}"
    aug_mask_filename = f"aug_{image_to_mask[target_img_filename]}"

    cv2.imwrite(os.path.join(OUTPUT_PATH, aug_img_filename), target_image)
    cv2.imwrite(os.path.join(OUTPUT_PATH, aug_mask_filename), target_mask)

print(f"Augmented dataset saved in '{OUTPUT_PATH}'")
