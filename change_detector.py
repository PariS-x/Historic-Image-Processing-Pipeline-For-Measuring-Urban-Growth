import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 

CHANGE_THRESHOLD = 70 
KERNEL_SIZE = (9, 9)

print("Starting Satellite Change Analysis...")

# --- Helper function to find and load images ---
def load_image_robust(base_filename):
    """Helper function to load an image, trying common extensions."""
    extensions_to_try = ['.png', '.jpg', '.jpeg']
    
    for ext in extensions_to_try:
        filename = base_filename + ext
        if os.path.exists(filename):
            img = cv2.imread(filename)
            if img is not None:
                print(f"Successfully loaded '{filename}'")
                return img, filename
            else:
                print(f"Found '{filename}' but could not read it. Check file integrity.")
                return None, filename
    
    return None, base_filename

# --- Step 1: Load & Pre-process (Exp 1, 4) ---

# Load images
img_before, before_name = load_image_robust('before')
img_after, after_name = load_image_robust('after')

# Check if images were loaded successfully
if img_before is None or img_after is None:
    print("-------------------------------------------------")
    print("FATAL ERROR: Could not load images.")
    print(f"Current working directory is: {os.getcwd()}")
    print("Please make sure you have files named 'before.png' (or .jpg/.jpeg)")
    print("and 'after.png' (or .jpg/.jpeg) in this *exact* folder.")
    print("And that you've completed Phase 1 (Data Acquisition) from the guide.")
    print("-------------------------------------------------")
    if img_before is None:
        print(f"Error: Could not find or read 'before.png/jpg/jpeg'.")
    if img_after is None:
        print(f"Error: Could not find or read 'after.png/jpg/jpeg'.")
    exit()

# Check if dimensions are *exactly* the same, as a final safeguard
if img_before.shape != img_after.shape:
    print("-------------------------------------------------")
    print("FATAL ERROR: Image dimensions do not match.")
    print(f"  {before_name} is {img_before.shape[1]}x{img_before.shape[0]} pixels")
    print(f"  {after_name} is  {img_after.shape[1]}x{img_after.shape[0]} pixels")
    print("Please re-crop them to be the *exact* same size (Step 1.3 in the guide).")
    print("-------------------------------------------------")
    exit()

# Convert to grayscale (Exp 1)
gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

# (Exp 4)
# We use a Median Blur, which is great for "salt-and-pepper" sensor noise.
denoised_before = cv2.medianBlur(gray_before, 5)
denoised_after = cv2.medianBlur(gray_after, 5)

print("Step 1 Complete: Images loaded, grayscaled, and denoised.")

# --- Step 2: Normalize Contrast (Exp 2, 5) ---

# We apply Histogram Equalization (Exp 5) to both images.
# This stretches their contrast to the full 0-255 range,
# making them more comparable by adjusting for different lighting/haze.
norm_before = cv2.equalizeHist(denoised_before)
norm_after = cv2.equalizeHist(denoised_after)

print("Step 2 Complete: Contrast normalized via Histogram Equalization.")

# --- Step 3: Detect Pixel-wise Change (Exp 3) ---

# We find the *absolute difference* between the two normalized images.
# The resulting `diff_image` is a raw heatmap:
# Black (0) = no change
# Bright White (255) = maximum change
diff_image = cv2.absdiff(norm_before, norm_after)

print("Step 3 Complete: Absolute difference calculated.")

# --- Step 4: Isolate Significant Change (Thresholding) ---

# Convert our grayscale `diff_image` into a binary (black/white) "mask".
# We use our `CHANGE_THRESHOLD` value.
# Any pixel with a change value *less than* the threshold becomes 0 (black).
# Any pixel *above* the threshold becomes 255 (white).
_ , thresh_mask = cv2.threshold(diff_image, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)

print(f"Step 4 Complete: Threshold mask created at value {CHANGE_THRESHOLD}.")

# --- Step 5: Refine Change Regions (Exp 7) ---

# The `thresh_mask` is noisy. We must clean it.
# This is the *perfect* use for Morphological Operations.
kernel = np.ones(KERNEL_SIZE, np.uint8)

# 1. Opening: (Erode -> Dilate)
#    This "opens" small gaps, effectively *removing* small, isolated white dots (noise).
#
opening_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 2. Closing: (Dilate -> Erode)
#    This "closes" small gaps, effectively *filling* small black holes
#    inside our larger white change-areas.
#
final_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

print("Step 5 Complete: Mask refined with Morphological Operations.")

# --- Step 6: Visualize & Quantify (Add-ons & Exp 8) ---

print("Starting Step 6: Visualization and Quantification...")

# 1. Generate Heatmap (Add-on)
# Apply a color map to the *raw difference image* from Step 3 for a nice visual.
heatmap = cv2.applyColorMap(diff_image, cv2.COLORMAP_JET)

# 2. Find Outlines (Exp 8)
# Use the `final_mask` to find the *outlines* of all white (changed) areas.
# `cv2.RETR_EXTERNAL` finds only the outermost contours.
contours, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3. Create Overlay 
# We draw the red contours onto a *copy* of the original color "after" image.
overlay_image = img_after.copy()
cv2.drawContours(overlay_image, contours, -1, (0, 0, 255), 2) # (Color BGR, Thickness)

# 4. Compute Percentage 
# How many pixels changed?
total_pixels = final_mask.shape[0] * final_mask.shape[1]
change_pixels = cv2.countNonZero(final_mask)
percent_changed = (change_pixels / total_pixels) * 100

print("--- Analysis Complete ---")
print(f"Percentage of area changed: {percent_changed:.2f}%")
print("-------------------------")

# --- Step 7: Display Final Report (Using Matplotlib) ---

print("Displaying final report... Close the new window to exit.")

# We will create a 2x3 grid of images to show our results.
plt.figure(figsize=(20, 10))

# Panel 1: Before
plt.subplot(2, 3, 1)
plt.title('Before (Normalized)')
plt.imshow(norm_before, cmap='gray')

# Panel 2: After
plt.subplot(2, 3, 2)
plt.title('After (Normalized)')
plt.imshow(norm_after, cmap='gray')

# Panel 3: Heatmap
plt.subplot(2, 3, 3)
plt.title('Change Heatmap')
# Matplotlib expects RGB images, but OpenCV uses BGR. We must convert.
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

# Panel 4: Raw Difference
plt.subplot(2, 3, 4)
plt.title('Raw Difference')
plt.imshow(diff_image, cmap='gray')

# Panel 5: Final Mask
plt.subplot(2, 3, 5)
plt.title('Final Change Mask')
plt.imshow(final_mask, cmap='gray')

# Panel 6: Overlay
plt.subplot(2, 3, 6)
plt.title('Change Overlay')
# Convert BGR to RGB for Matplotlib
plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))

# Add a main title to the whole figure
plt.suptitle(f'Environmental Change Analysis ({percent_changed:.2f}% Changed)', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle


plt.show()




