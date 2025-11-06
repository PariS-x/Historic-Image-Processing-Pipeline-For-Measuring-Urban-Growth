import cv2
import numpy as np
import os

# --- Configuration ---
# These are our final, tuned parameters
CHANGE_THRESHOLD = 70
KERNEL_SIZE = (9, 9)

# Input files
BEFORE_FILE = 'before.jpg'
AFTER_FILE = 'after.jpg'

# Output directory
OUTPUT_DIR = "output_proof"

# --- Setup ---
print(f"Starting Experiment Proof Generation...")
print(f"Reading '{BEFORE_FILE}' and '{AFTER_FILE}'...")

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving all intermediate steps to '/{OUTPUT_DIR}' folder.\n")

# --- Load Images ---
img_before_raw = cv2.imread(BEFORE_FILE)
img_after_raw = cv2.imread(AFTER_FILE)

if img_before_raw is None or img_after_raw is None:
    print(f"FATAL ERROR: Could not load images.")
    print(f"Please make sure '{BEFORE_FILE}' and '{AFTER_FILE}' are in this folder.")
    exit()

# --- Automated Registration (Resize) ---
# Ensure both images are the exact same size for pixel-math
target_w = min(img_before_raw.shape[1], img_after_raw.shape[1])
target_h = min(img_before_raw.shape[0], img_after_raw.shape[0])
dsize = (target_w, target_h)

img_before = cv2.resize(img_before_raw, dsize, interpolation=cv2.INTER_AREA)
img_after = cv2.resize(img_after_raw, dsize, interpolation=cv2.INTER_AREA)
print(f"Step 0: Images resized to {dsize} for registration.")
cv2.imwrite(os.path.join(OUTPUT_DIR, "0_registered_before.png"), img_before)
cv2.imwrite(os.path.join(OUTPUT_DIR, "0_registered_after.png"), img_after)

# ===================================================================
# --- ✅ Aim 1 (Part 1): Image Basics (Grayscale) ---
# ===================================================================
print("✅ Applying Aim 1 (Part 1): Grayscale Conversion...")
gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(OUTPUT_DIR, "1_aim_1_grayscale_before.png"), gray_before)
cv2.imwrite(os.path.join(OUTPUT_DIR, "1_aim_1_grayscale_after.png"), gray_after)

# ===================================================================
# --- ✅ Aim 4: Noise Removal ---
# ===================================================================
print("✅ Applying Aim 4: Noise Removal (Median Blur)...")
denoised_before = cv2.medianBlur(gray_before, 5)
denoised_after = cv2.medianBlur(gray_after, 5)
cv2.imwrite(os.path.join(OUTPUT_DIR, "2_aim_4_denoised_before.png"), denoised_before)
cv2.imwrite(os.path.join(OUTPUT_DIR, "2_aim_4_denoised_after.png"), denoised_after)

# ===================================================================
# --- ✅ Aim 5: Histogram Techniques ---
# ===================================================================
print("✅ Applying Aim 5: Histogram Equalization...")
norm_before = cv2.equalizeHist(denoised_before)
norm_after = cv2.equalizeHist(denoised_after)
cv2.imwrite(os.path.join(OUTPUT_DIR, "3_aim_5_normalized_before.png"), norm_before)
cv2.imwrite(os.path.join(OUTPUT_DIR, "3_aim_5_normalized_after.png"), norm_after)

# ===================================================================
# --- ✅ Aim 3 & Aim 1 (Part 2): Compare Two Images (Math Ops) ---
# ===================================================================
print("✅ Applying Aim 3 & Aim 1 (Part 2): Image Differencing (absdiff)...")
diff_image = cv2.absdiff(norm_before, norm_after)
cv2.imwrite(os.path.join(OUTPUT_DIR, "4_aim_3_raw_difference.png"), diff_image)

# ===================================================================
# --- ✅ Aim 2: Point Processing ---
# ===================================================================
print(f"✅ Applying Aim 2: Point Processing (Threshold at {CHANGE_THRESHOLD})...")
_ , thresh_mask = cv2.threshold(diff_image, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(OUTPUT_DIR, "5_aim_2_threshold_mask_raw.png"), thresh_mask)

# ===================================================================
# --- ✅ Aim 7: Morphological Operations ---
# ===================================================================
print(f"✅ Applying Aim 7: Morphological Ops (Kernel={KERNEL_SIZE})...")
kernel = np.ones(KERNEL_SIZE, np.uint8)

# Part 1: Opening (Removes salt noise)
opening_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)
print("  - Step 7a: MORPH_OPEN (Removes white noise dots)")
cv2.imwrite(os.path.join(OUTPUT_DIR, "6_aim_7a_morph_open.png"), opening_mask)

# Part 2: Closing (Fills pepper holes)
final_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
print("  - Step 7b: MORPH_CLOSE (Fills black holes)")
cv2.imwrite(os.path.join(OUTPUT_DIR, "7_aim_7b_final_mask.png"), final_mask)

# ===================================================================
# --- ✅ Aim 10: Region-Based Segmentation ---
# ===================================================================
print("✅ Applying Aim 10: Region Segmentation...")
print("  - The 'final_mask.png' *is* the result of segmentation.")
# No new image, the 'final_mask' is the proof.

# ===================================================================
# --- ✅ Aim 8: Edge Detection ---
# ===================================================================
print("✅ Applying Aim 8: Edge Detection (findContours)...")
overlay_image = img_after.copy()
# Find the *edges* of the segmented regions
contours, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw the edges in red
cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2) # BGR
cv2.imwrite(os.path.join(OUTPUT_DIR, "8_aim_8_final_overlay.png"), overlay_image)

# ===================================================================
# --- Final Quantification ---
# ===================================================================
total_pixels = final_mask.shape[0] * final_mask.shape[1]
change_pixels = cv2.countNonZero(final_mask)
percent_changed = (change_pixels / total_pixels) * 100

print("\n--- Analysis Complete ---")
print(f"Percentage of area changed: {percent_changed:.2f}%")
print(f"All 9 proof images have been saved to the '/{OUTPUT_DIR}' folder.")
print("-------------------------")
