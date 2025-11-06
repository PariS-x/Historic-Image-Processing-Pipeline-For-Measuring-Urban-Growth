import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Configuration ---
# These are our final, tuned parameters
CHANGE_THRESHOLD = 70
KERNEL_SIZE = (9, 9)

# Input files
BEFORE_FILE = 'before.jpg'
AFTER_FILE = 'after.jpg'

# --- Setup ---
print(f"Starting Experiment Proof Generation...")
print(f"Reading '{BEFORE_FILE}' and '{AFTER_FILE}'...")

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

# ===================================================================
# --- ✅ Aim 1 (Part 1): Image Basics (Grayscale) ---
# ===================================================================
print("✅ Applying Aim 1 (Part 1): Grayscale Conversion...")
gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

# ===================================================================
# --- ✅ Aim 4: Noise Removal ---
# ===================================================================
print("✅ Applying Aim 4: Noise Removal (Median Blur)...")
denoised_before = cv2.medianBlur(gray_before, 5)
denoised_after = cv2.medianBlur(gray_after, 5)

# ===================================================================
# --- ✅ Aim 5: Histogram Techniques ---
# ===================================================================
print("✅ Applying Aim 5: Histogram Equalization...")
norm_before = cv2.equalizeHist(denoised_before)
norm_after = cv2.equalizeHist(denoised_after)

# ===================================================================
# --- ✅ Aim 3 & Aim 1 (Part 2): Compare Two Images (Math Ops) ---
# ===================================================================
print("✅ Applying Aim 3 & Aim 1 (Part 2): Image Differencing (absdiff)...")
diff_image = cv2.absdiff(norm_before, norm_after)

# ===================================================================
# --- ✅ Aim 2: Point Processing ---
# ===================================================================
print(f"✅ Applying Aim 2: Point Processing (Threshold at {CHANGE_THRESHOLD})...")
_ , thresh_mask = cv2.threshold(diff_image, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)

# ===================================================================
# --- ✅ Aim 7: Morphological Operations ---
# ===================================================================
print(f"✅ Applying Aim 7: Morphological Ops (Kernel={KERNEL_SIZE})...")
kernel = np.ones(KERNEL_SIZE, np.uint8)

# Part 1: Opening (Removes salt noise)
opening_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)
print("  - Step 7a: MORPH_OPEN (Removes white noise dots)")

# Part 2: Closing (Fills pepper holes)
final_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
print("  - Step 7b: MORPH_CLOSE (Fills black holes)")

# ===================================================================
# --- ✅ Aim 10: Region-Based Segmentation ---
# ===================================================================
print("✅ Applying Aim 10: Region Segmentation...")
print("  - The 'Final Mask' *is* the result of segmentation.")

# ===================================================================
# --- ✅ Aim 8: Edge Detection ---
# ===================================================================
print("✅ Applying Aim 8: Edge Detection (findContours)...")
overlay_image = img_after.copy()
# Find the *edges* of the segmented regions
contours, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw the edges in red
cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2) # BGR
print("  - Contours found and drawn.")

# ===================================================================
# --- Final Quantification & Visualization ---
# ===================================================================
total_pixels = final_mask.shape[0] * final_mask.shape[1]
change_pixels = cv2.countNonZero(final_mask)
percent_changed = (change_pixels / total_pixels) * 100

print("\n--- Analysis Complete ---")
print(f"Percentage of area changed: {percent_changed:.2f}%")
print("Displaying proof plot window... Close the window to exit.")
print("-------------------------")

# --- Create the Matplotlib Figure ---
# UPDATED: Changed to a 3x3 grid for a cleaner layout
fig, axs = plt.subplots(3, 3, figsize=(16, 16), facecolor="#2d3748")
fig.suptitle(f'Experiment Proofs: Step-by-Step Pipeline ({percent_changed:.2f}% Change)', fontsize=22, color="cyan")

# Set a consistent theme
plt.style.use('dark_background')
title_color = "cyan"
title_fontsize = 14

# --- Row 1: Pre-processing ---
axs[0, 0].imshow(gray_after, cmap='gray')
axs[0, 0].set_title('Exp 1: Grayscale', color=title_color, fontsize=title_fontsize)

axs[0, 1].imshow(denoised_after, cmap='gray')
axs[0, 1].set_title('Exp 4: Denoised', color=title_color, fontsize=title_fontsize)

axs[0, 2].imshow(norm_after, cmap='gray')
axs[0, 2].set_title('Exp 5: Normalized', color=title_color, fontsize=title_fontsize)

# --- Row 2: Change Detection ---
axs[1, 0].imshow(diff_image, cmap='gray')
axs[1, 0].set_title('Exp 3: Raw Difference', color=title_color, fontsize=title_fontsize)

axs[1, 1].imshow(thresh_mask, cmap='gray')
axs[1, 1].set_title(f'Exp 2: Raw Threshold (T={CHANGE_THRESHOLD})', color=title_color, fontsize=title_fontsize)

axs[1, 2].imshow(opening_mask, cmap='gray')
axs[1, 2].set_title(f'Exp 7a: Morphological Open', color=title_color, fontsize=title_fontsize)

# --- Row 3: Final Results ---
axs[2, 0].imshow(final_mask, cmap='gray')
axs[2, 0].set_title('Exp 7b/10: Final Mask (Segmentation)', color=title_color, fontsize=title_fontsize)

axs[2, 1].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
axs[2, 1].set_title('Exp 8: Final Overlay (Edges)', color=title_color, fontsize=title_fontsize)

axs[2, 2].imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
axs[2, 2].set_title('Original "After" Image (Context)', color=title_color, fontsize=title_fontsize)

# Hide all axes ticks
for ax in axs.flat:
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.show()

